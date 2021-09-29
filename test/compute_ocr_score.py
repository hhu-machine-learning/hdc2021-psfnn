from OCR_evaluation import evaluateImage as compute_ocr_score
import os, glob
import numpy as np
import multiprocessing.pool

def main():
    step = 4
    font = "Times"

    input_dir = f"../tmp/step{step}/{font}"
    txt_dir = os.path.expanduser(f"~/data/hdc2021/step{step}/{font}/CAM02")

    png_paths = sorted(glob.glob(os.path.join(input_dir, "**/*.png"), recursive=True))

    ocr_scores = []

    # Might want to limit number of concurrent processes if RAM is scarce
    with multiprocessing.pool.ThreadPool(processes=None) as pool:

        for png_path in png_paths:
            # Skip non-text images
            if any(png_path.endswith(suffix) for suffix in [
                "_LSF_X.png",
                "_LSF_Y.png",
                "_PSF.png",
            ]): continue

            print("processing", png_path)

            _, filename = os.path.split(png_path)

            txt_path = os.path.join(txt_dir, filename.replace(".png", ".txt"))

            ocr_scores.append(pool.apply_async(compute_ocr_score, args=(png_path, txt_path)))

        ocr_scores = [score.get() for score in ocr_scores]
        # Replace invalid scores with 0.0
        ocr_scores = [0.0 if score is None else score for score in ocr_scores]

    pool.close()
    pool.join()

    print()
    mean_score = np.mean(ocr_scores)
    print(f"Step {step} - {font} Font - OCR score {mean_score}")
    print()

if __name__ == "__main__":
    main()
