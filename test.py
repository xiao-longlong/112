import cv2

def draw_yolo_annotations_on_image(image, annotations, output_path):

    # 获取图像的宽度和高度
    height, width, _ = image.shape

    # 绘制标注框
    for ann in annotations:
        class_id, center_x, center_y, bbox_width, bbox_height = ann
        # 计算实际的边界框坐标
        x1 = int((center_x - bbox_width / 2) * width)
        y1 = int((center_y - bbox_height / 2) * height)
        x2 = int((center_x + bbox_width / 2) * width)
        y2 = int((center_y + bbox_height / 2) * height)

        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 可以选择在框上标注类别 ID
        cv2.putText(image, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # 保存绘制后的图像
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to {output_path}")

if __name__ == "__main__":
    image = cv2.imread("/home/wxl/wxlcode/Python-Multiple-Image-Stitching/images/3.jpg")
    annotations = [[0,0.9,0.9,0.1,0.1]]
    draw_yolo_annotations_on_image(image, annotations, "test_annotated.jpg")
