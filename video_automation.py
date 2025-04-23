import argparse
from pathlib import Path
import subprocess

def run_inference(script_path, args):
    cmd = ["python", str(script_path)] + args
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run all CV inferences on a video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--nervous-model", required=False, help="Path to nervous model .pth")
    parser.add_argument("--eye-model", required=False, help="Path to eye contact model .pkl")
    parser.add_argument("--output-dir", required=False, help="Directory to save output JSON files", default="output")

    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Paths to utils
    base = Path(__file__).resolve().parents[0] / "src" / "video_analysis"
    nervous_script = base / "infer_nervous.py"
    eye_script = base / "infer_eye.py"
    motion_script = base / "infer_background.py"

    # Run each module
    # print("Nervousness analysis...")
    # run_inference(nervous_script, [
    #     "--input-video", str(video_path)
    # ])

    print("Eye contact analysis...")
    run_inference(eye_script, [
        "--video", str(video_path)
    ])

    print("Motion analysis...")
    run_inference(motion_script, [
        "--input-video", str(video_path)
    ])

    print(f"[✓] All analysis completed. Results saved to: {output_dir}")

if __name__ == "__main__":
    main()