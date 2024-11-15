import os
import shutil


class Fake:
    def generate_same_images_from_source(self, src_image: str, dst_folder: str, number: int):
        os.makedirs(dst_folder, exist_ok=True)  # 创建目标文件夹，如果已存在则忽略

        # 复制图片并重命名
        for i in range(1, number + 1):
            dst_image = os.path.join(dst_folder, f"img_{i}.jpg")
            shutil.copy(src_image, dst_image)
            print(f"Copied {src_image} to {dst_image}")

if __name__ == '__main__':
    fake = Fake()
    fake.generate_same_images_from_source("../datasets/img.jpg", dst_folder="../datasets/img", number=100)