import math
import cv2
import numpy as np
import copy
import torch
from skimage.morphology import skeletonize
from model import Utils

TERM = 10

class StrokeMaker:
    def __init__(self):
        self.utils = Utils()

    def preprocess(self,img):
        # --- 1. 細線化 ---
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        skel = skeletonize(bin_img).astype(np.uint8)

        # --- 2. 端点検出 ---
        endpoints = []
        h, w = skel.shape
        for y in range(1, h-1):
            for x in range(1, w-1):
                if skel[y, x] == 1:
                    neighborhood = skel[y-1:y+2, x-1:x+2]
                    count = np.sum(neighborhood) - 1
                    if count == 1:
                        endpoints.append((x, y))
        
        # --- 3. パス生成 ---
        used = [True for i in range(len(endpoints))]
        pathes = self.make_path(skel, endpoints, [], used) # パスを端点を総当たりで生成

        # --- 4. パス選択 ---
        chosen_path = self.choose_path(pathes) # パスを評価して1本に選択
        
        # --- 5. 結果の可視化 ---
        skel = (skel * 255).astype(np.uint8)
        skel = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        for (x, y) in endpoints:
            cv2.circle(skel, (x, y), 2, (0, 0, 255), -1)  # 赤: 端点
        for (x, y) in chosen_path:
            cv2.circle(skel, (x, y), 1, (0, 255, 0), -1)  # 緑: 選択されたパス

        return skel,chosen_path



    def make_path(self, skel, endpoints, stacks, used):
        return_path = []
        for i in range(len(endpoints)):
            if used[i]:
                used[i] = False
                stacks.append(endpoints[i])
                return_path += self.make_path(skel, endpoints, stacks, used)
                used[i] = True
                stacks.pop()
        if len(return_path) == 0:
            return_path.append([])
            for i in range(len(stacks)-1):
                start = stacks[i]
                end = stacks[i+1]
                return_path[0] += self.bfs(start, end, copy.deepcopy(skel))
        return return_path



    def bfs(self, start, end, skel):
        queue = [[start,[start]]]
        skel[start[1],start[0]] = 0
        while len(queue) > 0:
            element = queue.pop(0)
            x, y = element[0]
            path = copy.deepcopy(element[1])
            if (x, y) == end:
                path.append((x, y))
                return path
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if skel[y + dy, x + dx] == 1:
                        nx, ny = x + dx, y + dy
                        if abs(nx-path[-1][0])+abs(ny-path[-1][1]) > TERM:
                            path.append((nx, ny))
                        skel[ny, nx] = 0
                        queue.append([[nx, ny], path])
        return []



    def choose_path(self, paths):
        len_score = np.array([len(p) for p in paths])
        len_score = (np.min(len_score)/len_score)**2
        path = paths[np.argmax(len_score)]
        path = self.utils.sort_stroke(np.array(path))
        return path

def main():
    src = cv2.imread("otu.png") #---ソースファイルを選択---
    src = cv2.resize(src,(200,300))
    stroke_maker = StrokeMaker() # ストロークメーカークラスを作成
    dst = stroke_maker.preprocess(src)
    # preprocessメソッドに画像を入力すると
    # [ストロークの描画画像,ピクセル座標系のストローク配列(numpy.array型)]
    # の配列が返される。
    src += dst[0]
    cv2.imshow("src",src)
    #cv2.imshow("skel",dst[0])
    print("chosen_path:",dst[1])
    #print(len(dst[1]))

main()
cv2.waitKey(0)
cv2.destroyAllWindows()