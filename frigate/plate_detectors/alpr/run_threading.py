from alpr import AutoLPR
from video_decoder import VideoDecoder
from video_display import VideoDisplay


def main():
    # source = 'rtsp://root:12345678aA@192.168.0.248:554/live1s1.sdp'
    source = "rtsp://root:123456a@@192.168.0.243/axis-media/media.3gp"
    recog = AutoLPR()
    decoder = VideoDecoder(source).start()
    display = VideoDisplay(decoder.frame).start()

    while True:
        if decoder.stopped or display.stopped:
            decoder.stop()
            display.stop()
            break

        frame = decoder.frame
        frame = recog.processFrame(frame)
        display.frame = frame


if __name__ == '__main__':
    main()
