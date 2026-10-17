"""
Interactive homography helper.
Run: python compute_homography.py --video path/to/frame_image.jpg --out_homography H.npy
Click 4 image points in the shown window in this order:
 top-left, top-right, bottom-left, bottom-right (or any consistent order)
Then enter the corresponding real-world coordinates (X,Y in meters) for each clicked point.
The script computes image->ground homography and saves it to H.npy.
"""
import cv2, argparse, numpy as np
clicked = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked.append((x,y))
        print(f"Clicked: {(x,y)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to an image/frame to click points on')
    parser.add_argument('--out_homography', type=str, default='H.npy', help='Path to save homography .npy')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('frame', on_mouse)
    print("Click 4 points on the image. Press 'q' when done.")
    while True:
        disp = img.copy()
        for p in clicked:
            cv2.circle(disp, p, 6, (0,255,0), -1)
        cv2.imshow('frame', disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    cv2.destroyAllWindows()
    if len(clicked) < 4:
        print("Need 4 points. Exiting.")
        return
    image_pts = np.array(clicked[:4], dtype=np.float32)
    world_pts = []
    print("Enter real-world coordinates (X Y in meters) for each clicked point, in same order:")
    for i in range(4):
        s = input(f"Point {i+1} (format: X Y) > ")
        X,Y = map(float, s.strip().split())
        world_pts.append((X,Y))
    world_pts = np.array(world_pts, dtype=np.float32)
    H, status = cv2.findHomography(image_pts, world_pts)
    np.save(args.out_homography, H)
    print("Saved homography to", args.out_homography)

if __name__ == '__main__':
    main()
