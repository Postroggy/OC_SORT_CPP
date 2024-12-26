/*
 * note: 这里只是声明接口，用于与 C++ 层通过 JNI (Java Native Interface) 进行交互
 */
package com.gyq.yolov8;

import android.content.res.AssetManager;
import android.view.Surface;

public class Yolov8Ncnn {
    // mgr 是 AssetManager 用来访问应用的资源文件, modelid 是模型的 ID，cpugpu 指定使用 CPU 还是 GPU
    public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
    // facing 指定打开摄像头的方向
    public native boolean openCamera(int facing);
    // 关闭摄像头
    public native boolean closeCamera();
    // 数 surface 是输出图像的显示界面，是一个 SurfaceView 控件
    public native boolean setOutputWindow(Surface surface);

    // 通过JNI，这些声明成 public native 的方法可以用C++方法实现
    static {
        System.loadLibrary("yolov8ncnn");
    }
}
