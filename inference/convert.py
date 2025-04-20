import argparse
from inference.inference_pipeline import InferencePipeline
from common.audio_utils import save_audio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ae_ckpt', required=True)
    parser.add_argument('--vocoder_ckpt', required=True)
    parser.add_argument('--speaker_id', type=int, default=0)
    args = parser.parse_args()

    pipeline = InferencePipeline(args.ae_ckpt, args.vocoder_ckpt)
    audio = pipeline.convert(args.input, 22050, args.speaker_id)
    save_audio(args.output, audio.cpu().numpy(), 22050)

if __name__ == "__main__":
    main()