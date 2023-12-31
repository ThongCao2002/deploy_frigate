import { h } from 'preact';
import { useCallback, useEffect, useRef, useState } from 'preact/hooks';
import { useApiHost } from '../../api';
import { isNullOrUndefined } from '../../utils/objectUtils';

import 'video.js/dist/video-js.css';

import videojs from 'video.js';
import type Player from 'video.js/dist/types/player';

interface OnTimeUpdateEvent {
  timestamp: number;
  isPlaying: boolean;
}

interface HistoryVideoProps {
  id?: string;
  isPlaying: boolean;
  currentTime: number;
  onTimeUpdate?: (event: OnTimeUpdateEvent) => void;
  onPause: () => void;
  onPlay: () => void;
}

export const HistoryVideo = ({
  id,
  isPlaying: videoIsPlaying,
  currentTime,
  onTimeUpdate,
  onPause,
  onPlay,
}: HistoryVideoProps) => {
  const apiHost = useApiHost();
  const videoRef = useRef<HTMLVideoElement>(null);

  const [video, setVideo] = useState<Player>();

  useEffect(() => {
    let video: Player
    if (videoRef.current) {
      video = videojs(videoRef.current, {})
      setVideo(video)
    }
    () => video?.dispose()
  }, [videoRef]);

  useEffect(() => {
    if (!video) {
      return
    }


    if (!id) {
      video.pause()
      return
    }

    video.src({
      src: `${apiHost}vod/event/${id}/master.m3u8`,
      type: 'application/vnd.apple.mpegurl',
    });
    video.poster(`${apiHost}api/events/${id}/snapshot.jpg`);
    if (videoIsPlaying) {
      video.play();
    }
  }, [video, id, apiHost, videoIsPlaying]);

  useEffect(() => {
    const video = videoRef.current;
    const videoExists = !isNullOrUndefined(video);
    const hasSeeked = currentTime >= 0;
    if (video && videoExists && hasSeeked) {
      video.currentTime = currentTime;
    }
  }, [currentTime, videoRef]);

  const onTimeUpdateHandler = useCallback(
    (event: Event) => {
      const target = event.target as HTMLMediaElement;
      const timeUpdateEvent = {
        isPlaying: videoIsPlaying,
        timestamp: target.currentTime,
      };
      onTimeUpdate && onTimeUpdate(timeUpdateEvent);
    },
    [videoIsPlaying, onTimeUpdate]
  );

  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if (video && (video as any).readyState() >= 1) {
      if (videoIsPlaying) {
        video.play()
      } else {
        video.pause()
      }
    }
  }, [video, videoIsPlaying])

  const onLoad = useCallback(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if (video && (video as any).readyState() >= 1 && videoIsPlaying) {
      video.play()
    }
  }, [video, videoIsPlaying])

  return (
    <div data-vjs-player>
      <video
        ref={videoRef}
        onTimeUpdate={onTimeUpdateHandler}
        onLoadedMetadata={onLoad}
        onPause={onPause}
        onPlay={onPlay}
        className="video-js vjs-fluid"
        data-setup="{}"
      />
    </div>
  );
};
