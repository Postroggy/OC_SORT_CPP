package com.gyq.yolov8;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.Spinner;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;


public class MainActivity extends Activity implements SurfaceHolder.Callback {
    public static final int REQUEST_CAMERA = 100; // note: 状态请求码(常量)


    private Yolov8Ncnn yolov8ncnn = new Yolov8Ncnn();     // Yolov8Ncnn 对象，用于加载模型并进行推理处理
    private int facing = 0; // 标志摄像头的前置或者后置

    private Spinner spinnerModel; // 主界面上的Spinner控件，用于切换模型
    private int current_model = 0;
    private Spinner spinnerCPUGPU; // Spinner控件，用于切换CPU/GPU
    private int current_cpugpu = 0;

    private SurfaceView cameraView; // SurfaceView 控件，用于显示相机画面

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // 根据Xml初始化界面布局(R是自动生成的资源类，控制所有的Resources)
        setContentView(R.layout.main);
        // 设置保持屏幕常亮
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // 获取布局中的相机画面预览控件 (SurfaceView)
        cameraView = (SurfaceView) findViewById(R.id.cameraview);
        // 设置相机传输过来的数据格式为 RGBA_8888 (具体的可以根据手机的不同而更改，一般是这个)
        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        // TODO: 为 SurfaceView 设置回调接口
        cameraView.getHolder().addCallback(this);

        // 获取切换前置后置摄像头的Button控件
        Button buttonSwitchCamera = (Button) findViewById(R.id.buttonSwitchCamera);
        // 为点击摄像头的按钮写一个槽函数(事件)，实现切换摄像头功能
        buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                // 0->1, 1->0
                int new_facing = 1 - facing;
                yolov8ncnn.closeCamera();
                yolov8ncnn.openCamera(new_facing);
                facing = new_facing;
            }
        });

        // 获取并设置模型选择 Spinner 的监听器
        spinnerModel = (Spinner) findViewById(R.id.spinnerModel);
        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                if (position != current_model) {
                    current_model = position;
                    // 重新加载(新的)模型
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
            }
        });

        // 获取并设置 CPU/GPU 选择 Spinner 的监听器
        spinnerCPUGPU = (Spinner) findViewById(R.id.spinnerCPUGPU);
        spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                if (position != current_cpugpu) {
                    current_cpugpu = position;
                    reload(); // 同样的，切换了设备，模型也需要重载
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
            }
        });
        // 第一次启动，用默认参数加载模型
        reload();
    }

    private void reload() {
        // 加载模型
        boolean ret_init = yolov8ncnn.loadModel(getAssets(), current_model, current_cpugpu);
        if (!ret_init) {
            Log.e("MainActivity", "yolov8ncnn loadModel failed");
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        // 当 SurfaceView 的大小发生变化时调用此方法，设置输出的 Surface 为 Camera 的预览界面
        yolov8ncnn.setOutputWindow(holder.getSurface());
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
    }

    // note: 从后台重新回到该程序时应该做的
    @Override
    public void onResume() {
        super.onResume();

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }

        yolov8ncnn.openCamera(facing);
    }

    // note: 程序到后台时，关闭摄像头
    @Override
    public void onPause() {
        super.onPause();

        yolov8ncnn.closeCamera();
    }
}
