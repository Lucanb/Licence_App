import numpy as np
import os
from imageio.v2 import imread, imwrite

def forwardDiff(u):
    x = np.zeros([2] + list(u.shape))
    x[0, :-1, :] = u[1:, :] - u[:-1, :]
    x[1, :, :-1] = u[:, 1:] - u[:, :-1]
    return x

def getDivergence(p):
    shape = p.shape[1:]
    xS = np.zeros(shape)
    yS = np.zeros(shape)
    
    yS[1:-1] = p[0][1:-1, :] - p[0][0:-2, :]
    yS[0] = p[0][0]
    yS[-1] = -p[0][-2]
    
    xS[:, 1:-1] = p[1][:, 1:-1] - p[1][:, 0:-2]
    xS[:, 0] = p[1][:, 0]
    xS[:, -1] = -p[1][:, -2]
    
    return xS + yS

def getObjFuncVal(lbda, p, g):
    nchannels = p.shape[3]
    n = 0
    for c in range(nchannels):
        n += lbda * getDivergence(p[:, :, :, c]) - g[:, :, c]
    return np.sum(n * n)

def projectionSolve(img, lbda, error_tol, max_it, print_output):
    shape = img.shape
    nchannels = shape[2]
    g = img
    p = p0 = np.zeros((2,) + shape)
    it = 1
    ONE = np.ones(shape)
    t = 1 / 4.0
    
    while it < max_it:
        val_p0 = getObjFuncVal(lbda, p0, g)
        
        div_p = np.zeros(shape)
        p = np.zeros((2,) + shape)
        for c in range(nchannels):
            div_p[:, :, c] = getDivergence(p0[:, :, :, c])
            f1 = forwardDiff(div_p[:, :, c] - g[:, :, c] / lbda)
            p[:, :, :, c] += (p0[:, :, :, c] + t * f1) / (ONE[:, :, c] + t * np.sqrt((f1**2)).sum(axis=0))
        
        val_p = getObjFuncVal(lbda, p, g)
        p0 = p
        
        if print_output:
            print(f"iteration : {it} --- fn value: {val_p:.5f} --- differential: {val_p - val_p0:.5f}")
        
        if np.abs(val_p - val_p0) < error_tol:
            break
        it += 1
    
    return lbda * getDivergence(p)

def startDenoise(img, lbda, error_tol, max_it, print_output=False):
    if print_output:
        print("Executing Chambolle...")
    
    p = projectionSolve(img, lbda, error_tol, max_it, print_output)
    denoised_img = img - p
    return denoised_img

def multiChannelConv(img):
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    return img

def startPreprocessing(inputDir, outputDir, lbda=0.12, eps=1e-5, max_iterations=100):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    for filename in os.listdir(inputDir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filePath = os.path.join(inputDir, filename)
            noiseImg = imread(filePath, mode='L') / 255.0
            noiseImg = multiChannelConv(noiseImg)

            newImg = startDenoise(noiseImg, lbda, eps, max_iterations, True)

            outPath = os.path.join(outputDir, f'denoised_{filename}')
            imwrite(outPath, (newImg * 255).astype(np.uint8))
            
            print(f'Done: -- {filename} -- and saved to: -- {outPath} --')

if __name__ == "__main__":
    input = r'C:\Users\lucan\OneDrive\Desktop\RN_TESTS\preprocessing_tests\input'
    output = r'C:\Users\lucan\OneDrive\Desktop\RN_TESTS\preprocessing_tests\output'
    startPreprocessing(input, output)
