package com.example.flund.motion;

import java.util.ArrayList;
import java.util.Vector;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Point;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import static org.opencv.imgproc.Imgproc.boundingRect;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "com.example.flund.motion";
    private Mat mGray;
    int blur = 20;
    int[] theObject = {0, 0};
    Rect objectBoundingRectangle = new Rect(0, 0, 0, 0);
    private CameraBridgeViewBase mOpenCvCameraView;
    Point temp;                //first value for cycle checking
    int cycle = 0;              // value for cycle count
    int loopnum, cycletemp, diff;           //difference to ignore small movements

    DatabaseReference mRootRef = FirebaseDatabase.getInstance().getReference();
    DatabaseReference mCountRef = mRootRef.child("cycle");

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    // Load native library after(!) OpenCV initialization
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.main_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    protected void onStart() {
        super.onStart();

        mCountRef.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {

            }

            @Override
            public void onCancelled(DatabaseError databaseError) {

            }
        });
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_3_0, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();

    }

    private void resetVars() {}

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat(height, width, CvType.CV_8UC1);
        resetVars();
    }

    public void onCameraViewStopped() {
        mGray.release();
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mGray = inputFrame.rgba();

        Core.transpose(mGray, mGray);
        Imgproc.resize(mGray, mGray, mGray.size(), 0, 0, 0);
        Core.flip(mGray, mGray, 1);               //flip img

        Size s = new Size(blur, blur);          //blur size

        int N = 2;          // number of cyclic frame buffer used

        // ring image buffer
        Mat[] buf = null;
        int last = 0;
        int threshold = 170;

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Size size = mGray.size(); // get current frame size
        int i, idx1 = last, idx2;
        Mat thresh;             //threshold img

        if (buf == null || buf[0].width() != size.width || buf[0].height() != size.height) {
            if (buf == null) {
                buf = new Mat[N];
            }

            for (i = 0; i < N; i++) {
                if (buf[i] != null) {
                    buf[i].release();
                    buf[i] = null;
                }
                buf[i] = new Mat(size, CvType.CV_8UC1);
                buf[i] = Mat.zeros(size, CvType.CV_8UC1);
            }
        }                       // allocate images reallocate if frame size is changed

        Imgproc.cvtColor(mGray, buf[last], Imgproc.COLOR_BGR2GRAY);     // convert frame to gray scale

        // index of (last - (N-1))th frame
        idx2 = (last + 1) % N;
        last = idx2;

        thresh = buf[idx2];

        Core.absdiff(buf[idx1], buf[idx2], thresh);           // getting difference between frames

        Imgproc.threshold(thresh, thresh, threshold, 255, Imgproc.THRESH_BINARY);   //threshold it
        Imgproc.blur(thresh, thresh, s);                                             //blur to clean rough edges
        Imgproc.threshold(thresh, thresh, threshold, 255, Imgproc.THRESH_BINARY);   //threshold it again

        contours.clear();
        Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);


        Vector<MatOfPoint> largestContourVec = new Vector<>();
        largestContourVec.add(contours.get(contours.size() - 1));


        objectBoundingRectangle = boundingRect(largestContourVec.elementAt(0));
        int xpos = objectBoundingRectangle.x + objectBoundingRectangle.width / 2;
        int ypos = objectBoundingRectangle.y + objectBoundingRectangle.height / 2;          //bounding rectangle around the largest contour

        theObject[0] = xpos;
        theObject[1] = ypos;            //update objects positions

        int x = theObject[0];
        int y = theObject[1];           //make some temp x and y variables


        if (cycletemp == 0) {
            temp = new Point(x, y);
        }                               //check for first point

        Point box = new Point(x, y);
        int x1 = x + 200;
        int y1 = y + 200;
        Point box2 = new Point(x1, y1);        //box points



        if (contours.size() > 0) {
            Log.i(TAG, "targetDetected");
            diff =  x1 - x;            //get difference
            if (box.equals(temp)) {
                Log.i(TAG, "noMotion");
                cycletemp++;
            } else {
                Log.i(TAG, "Motion");
                if (diff > 210)
                {
                    cycle++;
                    mCountRef.setValue(cycle);
                    loopnum++;
                }
                else if (diff < -210)
                {
                    cycle++;
                    mCountRef.setValue(cycle);
                    loopnum++;
                }
                else
                    Log.i(TAG, "SmallMotion");
            }
        }
        else {
            Log.i(TAG, "targetnotDetected");
        }


        if (loopnum > 1){
            if (box.equals(box)){
                cycletemp = 0;
                loopnum = 0;
                Log.i(TAG, "reset");
            }
        }



        Imgproc.rectangle(mGray, box, box2, new Scalar(0, 255, 0));                     //draw box of motion
        Imgproc.putText(mGray, "Num of cycles:" + cycle, new Point(10, 30), 2, 1, new Scalar(255, 0, 0));
        Imgproc.putText(mGray, "temp:" + temp, new Point(10, 65), 2, 1, new Scalar(255, 0, 0));
        Imgproc.putText(mGray, "box:" + box, new Point(10, 110), 2, 1, new Scalar(255, 0, 0));
        Imgproc.putText(mGray, "diff:" + diff, new Point(10, 160), 2, 1, new Scalar(255, 0, 0));                //write on frame

        return mGray;

    }
}



