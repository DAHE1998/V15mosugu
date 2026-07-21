// frame_cutter_gpu.cpp — v1, 视频-only, 单路, 朴素稳定
//
// Compile:
//   g++ -o frame_cutter_gpu frame_cutter_gpu.cpp \
//     $(pkg-config --cflags --libs libavformat libavcodec libavutil) \
//     -std=c++17 -O2
//
// Run:
//   ./frame_cutter_gpu <cut.json>
//
// Pipeline:
//   NVDEC(CUDA) → frame index gate → NVENC(CUDA) → MP4
//
// 禁止:
//   - rawvideo 中间文件
//   - CPU frame copy / hwdownload
//   - Python 参与帧处理
//   - 多 encoder
//   - frame 路由表
//   - 音频 (v1 不做)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/error.h>
#include <libavutil/opt.h>
}

// ─── 数据结构 ───────────────────────────────────────────────────
struct Segment {
    int id;
    int start_frame;
    int end_frame;
};

struct VideoInfo {
    int width, height;
    double fps;
    int total_frames;
    AVCodecID codec_id;
};

// ─── 工具 ────────────────────────────────────────────────────────
static void check(int err, const char* msg) {
    if (err < 0) {
        char buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(err, buf, sizeof(buf));
        fprintf(stderr, "ERROR: %s: %s\n", msg, buf);
        exit(1);
    }
}

static void fatal(const char* msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

// ─── JSON 解析 ──────────────────────────────────────────────────
static std::string extract_str(const std::string& s, const char* key) {
    std::string k = std::string("\"") + key + "\"";
    auto p = s.find(k);
    if (p == std::string::npos) return "";
    p = s.find(':', p + k.size());
    if (p == std::string::npos) return "";
    p = s.find('"', p + 1);
    if (p == std::string::npos) return "";
    auto e = s.find('"', p + 1);
    if (e == std::string::npos) return "";
    return s.substr(p + 1, e - p - 1);
}


static int extract_int(const std::string& s, const char* key) {
    std::string v = extract_str(s, key);
    return v.empty() ? 0 : std::stoi(v);
}
static std::vector<Segment> parse_json(const char* path, std::string& video_path) {
    std::ifstream f(path);
    if (!f) fatal("Cannot open JSON file");
    std::string s((std::istreambuf_iterator<char>(f)), {});
    video_path = extract_str(s, "video");
    if (video_path.empty()) fatal("No 'video' field in JSON");

    std::vector<Segment> segs;
    auto pos = s.find("\"segments\"");
    if (pos != std::string::npos) {
        auto arr = s.find('[', pos);
        if (arr != std::string::npos) {
            size_t i = arr;
            while (true) {
                auto o = s.find('{', i);
                if (o == std::string::npos) break;
                auto c = s.find('}', o);
                if (c == std::string::npos) break;
                std::string obj = s.substr(o, c - o + 1);
                segs.push_back({
                    extract_int(obj, "id"),
                    extract_int(obj, "start_frame"),
                    extract_int(obj, "end_frame")
                });
                i = c + 1;
            }
        }
    }
    std::sort(segs.begin(), segs.end(),
              [](const Segment& a, const Segment& b) { return a.start_frame < b.start_frame; });
    if (segs.empty()) fatal("No segments");
    return segs;
}

// ─── 视频探测 ───────────────────────────────────────────────────
static VideoInfo probe_video(const char* path) {
    AVFormatContext* fmt = nullptr;
    check(avformat_open_input(&fmt, path, NULL, NULL), "open input");
    check(avformat_find_stream_info(fmt, NULL), "find stream info");
    int vid_idx = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    check(vid_idx, "find video stream");

    AVStream* st = fmt->streams[vid_idx];
    VideoInfo info;
    info.width = st->codecpar->width;
    info.height = st->codecpar->height;
    info.codec_id = st->codecpar->codec_id;
    info.total_frames = (int)st->nb_frames;

    if (st->avg_frame_rate.num > 0)
        info.fps = (double)st->avg_frame_rate.num / st->avg_frame_rate.den;
    else if (st->r_frame_rate.num > 0)
        info.fps = (double)st->r_frame_rate.num / st->r_frame_rate.den;
    else
        info.fps = 30.0;

    avformat_close_input(&fmt);
    return info;
}

// ─── CUDA device ────────────────────────────────────────────────
static AVBufferRef* create_cuda_device() {
    AVBufferRef* dev = nullptr;
    check(av_hwdevice_ctx_create(&dev, AV_HWDEVICE_TYPE_CUDA, "cuda", NULL, 0),
          "create CUDA device");
    return dev;
}

// ─── Decoder (NVDEC) ────────────────────────────────────────────
static AVCodecContext* open_decoder(const VideoInfo& info, AVBufferRef* hw_dev) {
    const char* dec_name = nullptr;
    if (info.codec_id == AV_CODEC_ID_H264) dec_name = "h264_cuvid";
    else if (info.codec_id == AV_CODEC_ID_HEVC) dec_name = "hevc_cuvid";
    else fatal("Unsupported codec for HW decode");

    AVCodec* dec = (AVCodec*)avcodec_find_decoder_by_name(dec_name);
    if (!dec) fatal("HW decoder not found");

    AVCodecContext* ctx = avcodec_alloc_context3(dec);
    ctx->hw_device_ctx = av_buffer_ref(hw_dev);
    check(avcodec_open2(ctx, dec, NULL), "open decoder");
    return ctx;
}

// ─── Encoder (NVENC) ────────────────────────────────────────────
static AVCodecContext* open_encoder(const VideoInfo& info, AVBufferRef* hw_dev) {
    const char* enc_name = nullptr;
    if (info.codec_id == AV_CODEC_ID_H264) enc_name = "h264_nvenc";
    else if (info.codec_id == AV_CODEC_ID_HEVC) enc_name = "hevc_nvenc";

    AVCodec* enc = (AVCodec*)avcodec_find_encoder_by_name(enc_name);
    if (!enc) fatal("NVENC encoder not found");

    AVCodecContext* ctx = avcodec_alloc_context3(enc);
    ctx->width = info.width;
    ctx->height = info.height;
    ctx->pix_fmt = AV_PIX_FMT_CUDA;
    ctx->time_base = (AVRational){1, (int)info.fps};
    ctx->framerate = (AVRational){(int)info.fps, 1};
    ctx->gop_size = 30;
    ctx->max_b_frames = 0;
    ctx->hw_device_ctx = av_buffer_ref(hw_dev);

    av_opt_set(ctx->priv_data, "preset", "p4", 0);
    av_opt_set_int(ctx->priv_data, "cq", 26, 0);
    check(avcodec_open2(ctx, enc, NULL), "open encoder");
    return ctx;
}

// ─── Segment 输出 ───────────────────────────────────────────────
struct SegOut {
    AVFormatContext* fmt;
    AVStream* stream;
    AVCodecContext* enc;
    int64_t next_pts;
};

static SegOut open_output(const char* path, AVCodecContext* enc) {
    SegOut out{};
    check(avformat_alloc_output_context2(&out.fmt, NULL, NULL, path), "alloc output");
    out.stream = avformat_new_stream(out.fmt, enc->codec);
    if (!out.stream) fatal("new stream");
    check(avcodec_parameters_from_context(out.stream->codecpar, enc), "copy params");
    out.stream->time_base = enc->time_base;
    check(avio_open(&out.fmt->pb, path, AVIO_FLAG_WRITE), "open output");
    check(avformat_write_header(out.fmt, NULL), "write header");
    out.enc = enc;
    out.next_pts = 0;
    return out;
}

// 编码一帧并 mux（在同一个函数内，单一出口）
static void encode_and_mux(SegOut& out, AVFrame* frame) {
    AVFrame* fc = av_frame_clone(frame);
    check(avcodec_send_frame(out.enc, fc), "send to encoder");
    av_frame_unref(fc);

    AVPacket* opkt = av_packet_alloc();
    while (avcodec_receive_packet(out.enc, opkt) >= 0) {
        opkt->pts = out.next_pts++;
        opkt->dts = opkt->pts;
        opkt->duration = 1;
        av_packet_rescale_ts(opkt, out.enc->time_base, out.stream->time_base);
        opkt->stream_index = out.stream->index;
        check(av_interleaved_write_frame(out.fmt, opkt), "write packet");
        av_packet_unref(opkt);
        opkt = av_packet_alloc();
    }
    av_packet_free(&opkt);
}

// 完成一个 segment：flush encoder → write trailer → close
static void finish_segment(SegOut& out) {
    check(avcodec_send_frame(out.enc, NULL), "flush encoder");
    AVPacket* opkt = av_packet_alloc();
    while (avcodec_receive_packet(out.enc, opkt) >= 0) {
        opkt->pts = out.next_pts++;
        opkt->dts = opkt->pts;
        opkt->duration = 1;
        av_packet_rescale_ts(opkt, out.enc->time_base, out.stream->time_base);
        opkt->stream_index = out.stream->index;
        check(av_interleaved_write_frame(out.fmt, opkt), "write packet");
        av_packet_unref(opkt);
        opkt = av_packet_alloc();
    }
    av_packet_free(&opkt);
    av_write_trailer(out.fmt);
    avio_closep(&out.fmt->pb);
    avformat_free_context(out.fmt);
    avcodec_free_context(&out.enc);
}

// ─── 主流程 ─────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <cut.json>\n", argv[0]);
        return 1;
    }

    av_log_set_level(AV_LOG_ERROR);

    // 1. 解析 JSON
    std::string video_path;
    auto segments = parse_json(argv[1], video_path);

    // 2. 探测视频
    auto info = probe_video(video_path.c_str());
    printf("Video: %dx%d @ %.3ffps, %d frames\n",
           info.width, info.height, info.fps, info.total_frames);
    printf("Segments: %d\n", (int)segments.size());

    // 3. CUDA device
    AVBufferRef* hw_dev = create_cuda_device();

    // 4. Decoder
    AVCodecContext* dec = open_decoder(info, hw_dev);

    // 5. 打开输入
    AVFormatContext* in_fmt = nullptr;
    check(avformat_open_input(&in_fmt, video_path.c_str(), NULL, NULL), "open input");
    check(avformat_find_stream_info(in_fmt, NULL), "find stream info");
    int vid_idx = av_find_best_stream(in_fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    check(vid_idx, "find video stream");

    // 6. 为每个 segment 创建 encoder + output
    std::vector<SegOut> outputs;
    outputs.reserve(segments.size());
    for (const auto& seg : segments) {
        AVCodecContext* enc = open_encoder(info, hw_dev);
        char path[256];
        snprintf(path, sizeof(path), "segment_%04d.mp4", seg.id);
        outputs.push_back(open_output(path, enc));
    }

    // 7. 单指针追踪当前 segment
    int seg_idx = 0;

    // 8. Decode loop
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    int64_t frame_index = 0;
    int64_t total_decoded = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (av_read_frame(in_fmt, pkt) >= 0) {
        if (pkt->stream_index != vid_idx) {
            av_packet_unref(pkt);
            continue;
        }

        check(avcodec_send_packet(dec, pkt), "send packet");
        av_packet_unref(pkt);

        while (avcodec_receive_frame(dec, frame) >= 0) {
            // 单指针：segments 已排序，frame_index 只增不减
            while (seg_idx < (int)segments.size() &&
                   frame_index > segments[seg_idx].end_frame) {
                // 结束当前 segment
                finish_segment(outputs[seg_idx]);
                seg_idx++;
            }

            if (seg_idx < (int)segments.size() &&
                frame_index >= segments[seg_idx].start_frame &&
                frame_index <= segments[seg_idx].end_frame) {
                // 编码并 mux
                encode_and_mux(outputs[seg_idx], frame);
            }
            // else: 帧不属于任何 segment，丢弃

            av_frame_unref(frame);
            frame_index++;
            total_decoded++;
        }

        // 进度
        if (total_decoded % 500 == 0) {
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - t0).count();
            printf("  %ld frames (%.1fx realtime)\r", total_decoded,
                   dt > 0 ? total_decoded / (dt * info.fps) : 0);
            fflush(stdout);
        }
    }
    printf("\nDecoded: %ld frames\n", total_decoded);

    // 9. Flush 最后一个 segment
    if (seg_idx < (int)segments.size()) {
        finish_segment(outputs[seg_idx]);
    }

    // 10. Cleanup
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&dec);
    avformat_close_input(&in_fmt);
    av_buffer_unref(&hw_dev);

    printf("Done! %d segments written.\n", (int)segments.size());
    return 0;
}
