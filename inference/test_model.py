

import argparse
from inference.inference_pipeline import InferencePipeline
from common.audio_utils import save_audio

def main():
    parser = argparse.ArgumentParser(description="Run voice conversion on user-supplied WAV files.")
    parser.add_argument("--input", required=True, help="Path to source input WAV file")
    parser.add_argument("--ref", required=True, help="Path to reference speaker WAV file")
    parser.add_argument("--output", required=True, help="Path to save converted output WAV file")
    parser.add_argument("--ae_ckpt", default="checkpoints/autoencoder.pth", help="Path to autoencoder checkpoint")
    parser.add_argument("--hifi_ckpt", default="checkpoints/hifigan.pt", help="Path to HiFi-GAN checkpoint")
    parser.add_argument("--device", default="cuda", help="Device to use for inference (cuda or cpu)")
    args = parser.parse_args()

    pipeline = InferencePipeline(args.ae_ckpt, args.hifi_ckpt, device=args.device)
    audio, sr = pipeline.convert(args.input, args.ref)
    save_audio(args.output, audio.numpy(), sr)
    print(f"✅ Inference complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()