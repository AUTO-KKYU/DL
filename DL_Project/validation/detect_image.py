import cv2
import numpy as np

def set_label(image, text, contour):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    baseline = 0

    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    rect = cv2.boundingRect(contour)

    pt = (rect[0] + ((rect[2] - text_size[0]) // 2), rect[1] + ((rect[3] + text_size[1]) // 2))
    cv2.rectangle(image, (pt[0], pt[1] - text_size[1]), (pt[0] + text_size[0], pt[1]), (200, 200, 200), cv2.FILLED)
    cv2.putText(image, text, pt, font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

def main():
    img_input = cv2.imread("C:\\Users\\dknjy\\.anaconda\\shape.jpg")
    if img_input is None:
        print("Could not open or find the image")
        return

    img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_result = img_input.copy()

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(approx)

        if abs(area) > 100:
            size = len(approx)

            if size % 2 == 0:
                cv2.polylines(img_result, [approx], True, (0, 255, 0), 3)
                for k in range(size - 1):
                    cv2.line(img_result, tuple(approx[k][0]), tuple(approx[k + 1][0]), (0, 255, 0), 3)
                for point in approx:
                    cv2.circle(img_result, tuple(point[0]), 3, (0, 0, 255), -1)
            else:
                cv2.polylines(img_result, [approx], True, (0, 255, 0), 3)
                for k in range(size - 1):
                    cv2.line(img_result, tuple(approx[k][0]), tuple(approx[k + 1][0]), (0, 255, 0), 3)
                for point in approx:
                    cv2.circle(img_result, tuple(point[0]), 3, (0, 0, 255), -1)

            if size == 3:
                set_label(img_result, "triangle", contour)
            elif size == 4 and cv2.isContourConvex(approx):
                set_label(img_result, "rectangle", contour)
            elif size == 5 and cv2.isContourConvex(approx):
                set_label(img_result, "pentagon", contour)
            elif size == 6 and cv2.isContourConvex(approx):
                set_label(img_result, "hexagon", contour)
            elif size == 10 and cv2.isContourConvex(approx):
                set_label(img_result, "decagon", contour)
            else:
                set_label(img_result, str(size), contour)

    cv2.imshow("input", img_input)
    cv2.imshow("result", img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
