using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using OpenCvSharp;

/// <summary>
/// Face detection sample by OpenCvSharp
/// </summary>
/// 
/// References
/// - [結局はOpenCVが最強だった話](https://younaship.com/2019/04/15/%e7%b5%90%e5%b1%80%e3%81%afopencv%e3%81%8c%e6%9c%80%e5%bc%b7%e3%81%a0%e3%81%a3%e3%81%9f%e8%a9%b1/)
/// - [OpenCvSharp](https://github.com/shimat/opencvsharp)
public class FaceAndEyeDetectionDriver : MonoBehaviour
{
    [SerializeField]
    private RawImage _previewRawImage = null;

    private WebCamTexture _webCamTexture = null;
    private Texture2D _previewTexture2D = null;
    private CascadeClassifier _cFace, _cEye;
    private Task _detectionTask = null;
    private const double FACTOR_SCALE_FACE = 1.01;
    private const double FACTOR_SCALE_EYE = 1.01;
    private const int MIN_SIZE = 5;

    private void Awake()
    {
        if (_previewRawImage == null)
        {
            Debug.LogError("WebCamTextureDriver._previewRawImage is required.");
            enabled = false;
            return;
        }

        _webCamTexture = new WebCamTexture(320, 240);
        _webCamTexture.Play();

        _previewTexture2D = CreateClearTexture2D(_webCamTexture.width, _webCamTexture.height);
        _previewRawImage.texture = _previewTexture2D;

        _cFace = new CascadeClassifier(string.Format(
            "{1}{0}OpenCvSharp{0}haarcascades{0}haarcascade_frontalface_alt.xml",
                Path.DirectorySeparatorChar, Application.dataPath));
        _cEye = new CascadeClassifier(string.Format(
            "{1}{0}OpenCvSharp{0}haarcascades{0}haarcascade_eye.xml",
                Path.DirectorySeparatorChar, Application.dataPath)); 
    }

    private void Update()
    {
        if (_detectionTask != null && !_detectionTask.IsCompleted) return;

        Color32[] pixels32 = _webCamTexture.GetPixels32();
        int width = _webCamTexture.width, height = _webCamTexture.height;

        _detectionTask = Task.Run(() => {
            Mat baseMat = ToMat8UC3(pixels32, width, height);
            Mat grayMat = baseMat.CvtColor(ColorConversionCodes.BGR2GRAY);

            foreach (OpenCvSharp.Rect rect in _cFace.DetectMultiScale(grayMat, FACTOR_SCALE_FACE, MIN_SIZE))
            {
                var baseMatIn = grayMat[rect.Y, rect.Y + rect.Height, rect.X, rect.X + rect.Width];
                grayMat.Rectangle(rect, Scalar.White, 2);

                foreach (OpenCvSharp.Rect inRect in _cEye.DetectMultiScale(baseMatIn, FACTOR_SCALE_EYE, MIN_SIZE))
                {
                    Point fPoint = new Point(rect.X + inRect.X, rect.Y + inRect.Y);
                    Point ePoint = new Point(rect.X + inRect.X + inRect.Width, rect.Y + inRect.Y + inRect.Height);
                    grayMat.Rectangle(fPoint, ePoint, Scalar.White, 1);
                }
            }

            byte[] grayMatBytes = null;
            grayMat.GetArray<byte>(out grayMatBytes);
            Color32[] grayMatPixels = new Color32[grayMatBytes.Length];
            for (int index = 0; index < grayMatBytes.Length; index++)
                grayMatPixels[index] = new Color32(
                    grayMatBytes[index], grayMatBytes[index], grayMatBytes[index], byte.MaxValue);

            return grayMatPixels;
        }).ContinueWith(task => {
            _previewTexture2D.SetPixels32(task.Result);
            _previewTexture2D.Apply();

        }, TaskScheduler.FromCurrentSynchronizationContext());
    }

    private void OnDestroy()
    {
        if (_webCamTexture != null && _webCamTexture.isPlaying) _webCamTexture.Stop();
    }

    private Texture2D CreateClearTexture2D(int width, int height)
    {
        Texture2D texture = new Texture2D(width, height);
        Color32[] pixels = texture.GetPixels32();
        for (int x = 0; x < pixels.Length; x++) pixels[x] = Color.clear;
        texture.SetPixels32(pixels);
        texture.Apply();
        return texture;
    }

    private static Mat ToMat8UC3(Color32[] c32, int width, int height)
    {
        Mat mat = new Mat(height, width, MatType.CV_8UC3);
        int x_ = 0, y_ = height - 1;
        for (int i = 0; i < width * height; i++, x_++)
        {
            if (x_ >= width)
            {
                x_ = 0;
                y_--;
            }
            Vec3b v3b = new Vec3b(c32[i].r, c32[i].g, c32[i].b);
            mat.Set<Vec3b>(y_, x_, v3b);
        }
        return mat;
    }
}
