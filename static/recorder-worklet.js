/**
 * PCM recorder AudioWorklet processor.
 * Forwards raw Float32 mono frames from the render graph to the main thread.
 */
class PcmRecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) {
      // copy: the underlying buffer is reused by the audio thread
      this.port.postMessage(new Float32Array(channel));
    }
    return true;
  }
}

registerProcessor("pcm-recorder", PcmRecorderProcessor);
