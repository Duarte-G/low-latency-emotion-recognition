using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using System;
using TMPro;

public class EmotionDetectionSystem : MonoBehaviour
{
    [Header("Webcam Settings")]
    public RawImage webcamDisplay;
    public AspectRatioFitter aspectFitter;
    private WebCamTexture webcamTexture;
    private Texture2D capturedTexture;

    [Header("API Settings")]
    public string apiUrl = "http://localhost:5000/predict";
    public float checkInterval = 0.5f;

    [Header("Emoji Display")]
    public GameObject emojiPanel;
    public Image emojiImage;
    public Sprite neutralEmoji;
    public Sprite happyEmoji;
    public Sprite sadEmoji;
    public Sprite angryEmoji;

    [Header("Detection Control")]
    public KeyCode detectionKey = KeyCode.Space;

    private bool isDetecting = false;
    private string currentEmotion = "Neutral";
    private float currentConfidence = 0f;
    private bool isProcessing = false;
    private bool hasDetectedEmotion = false;

    [Header("Center Display")]
    public TextMeshProUGUI emotionCenterText;
    public GameObject detectionIndicator;

    void Start()
    {
        // Inicializar e iniciar a webcam
        webcamTexture = new WebCamTexture(640, 360, 15);
        if (webcamDisplay != null)
        {
            webcamDisplay.texture = webcamTexture;
            webcamDisplay.gameObject.SetActive(true); // Sempre visivel
        }

        // Iniciar a webcam
        webcamTexture.Play();
        // UpdateWebcamAspectRatio();

        // Configurar HUD
        // SetupHUDLayout();
        SetupCenterDisplay();

        // Iniciar coroutine para verificar emocoes
        StartCoroutine(CheckEmotionPeriodically());
    }

    void Update()
    {
        // Controle por tecla
        if (Input.GetKeyDown(detectionKey))
        {
            ToggleDetection();
        }

        // Atualizar emoji baseado na emocao atual
        UpdateEmojiDisplay();
    }

    void ToggleDetection()
    {
        isDetecting = !isDetecting;

        if (isDetecting)
        {
            // Apenas ativar a deteccao, a webcam ja esta visivel
            Debug.Log("Deteccao iniciada");
        }
        else
        {
            // Apenas desativar a deteccao, manter a webcam visivel
            Debug.Log("Deteccao parada - Emocao mantida: " + currentEmotion);
        }
    }

    void SetupHUDLayout()
    {
        // Procura o Canvas pai para obter o scaleFactor
        Canvas rootCanvas = GetComponentInParent<Canvas>();
        float scale = (rootCanvas != null) ? rootCanvas.scaleFactor : 1f;

        // Tamanhos e margens em PIXELS
        Vector2 webcamSizePx = new Vector2(300f, 200f);
        Vector2 webcamMarginPx = new Vector2(10f, 10f);

        Vector2 emojiSizePx = new Vector2(100f, 100f);
        Vector2 emojiMarginPx = new Vector2(10f, 220f);

        if (webcamDisplay != null)
        {
            RectTransform webcamRect = webcamDisplay.GetComponent<RectTransform>();
            // fixa no canto superior direito
            webcamRect.anchorMin = new Vector2(1f, 1f);
            webcamRect.anchorMax = new Vector2(1f, 1f);
            webcamRect.pivot = new Vector2(1f, 1f);

            // ajusta tamanho e posição levando em conta o scaleFactor do canvas
            webcamRect.sizeDelta = webcamSizePx / scale;
            webcamRect.anchoredPosition = new Vector2(-webcamMarginPx.x / scale, -webcamMarginPx.y / scale);
        }

        if (emojiPanel != null)
        {
            RectTransform emojiRect = emojiPanel.GetComponent<RectTransform>();
            emojiRect.anchorMin = new Vector2(1f, 1f);
            emojiRect.anchorMax = new Vector2(1f, 1f);
            emojiRect.pivot = new Vector2(1f, 1f);

            emojiRect.sizeDelta = emojiSizePx / scale;
            emojiRect.anchoredPosition = new Vector2(-emojiMarginPx.x / scale, -emojiMarginPx.y / scale);
        }
    }

    void UpdateWebcamAspectRatio()
    {
        if (webcamTexture != null && webcamTexture.isPlaying && aspectFitter != null)
        {
            float ratio = (float)webcamTexture.width / (float)webcamTexture.height;
            aspectFitter.aspectRatio = ratio;

            // Corrigir rotacao se necessario
            webcamDisplay.transform.localEulerAngles = new Vector3(0, 0, -webcamTexture.videoRotationAngle);

            // Espelhar a imagem se a webcam for frontal
            if (webcamTexture.videoVerticallyMirrored)
            {
                webcamDisplay.transform.localScale = new Vector3(-1, 1, 1);
            }
            else
            {
                webcamDisplay.transform.localScale = new Vector3(1, 1, 1);
            }
        }
    }

    IEnumerator CheckEmotionPeriodically()
    {
        while (true)
        {
            if (isDetecting && webcamTexture != null && webcamTexture.isPlaying && !isProcessing)
            {
                yield return StartCoroutine(CaptureAndSendFrame());
            }
            yield return new WaitForSeconds(checkInterval);
        }
    }

    IEnumerator CaptureAndSendFrame()
    {
        isProcessing = true;
        byte[] imageBytes = null;

        try
        {
            // Capturar o frame atual da webcam
            if (capturedTexture == null ||
                capturedTexture.width != webcamTexture.width ||
                capturedTexture.height != webcamTexture.height)
            {
                capturedTexture = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.RGB24, false);
            }

            capturedTexture.SetPixels(webcamTexture.GetPixels());
            capturedTexture.Apply();

            // Converter para JPG
            imageBytes = capturedTexture.EncodeToJPG();
        }
        catch (Exception e)
        {
            Debug.LogError("Erro ao capturar frame: " + e.Message);
            isProcessing = false;
            yield break;
        }

        // Criar formulario para enviar a imagem
        WWWForm form = new WWWForm();
        form.AddBinaryData("image", imageBytes, "frame.jpg", "image/jpeg");

        // Enviar para a API
        using (UnityWebRequest request = UnityWebRequest.Post(apiUrl, form))
        {
            request.downloadHandler = new DownloadHandlerBuffer();
            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                string json = request.downloadHandler.text;
                EmotionData data = JsonUtility.FromJson<EmotionData>(json);

                if (data != null)
                {
                    currentEmotion = data.emotion;
                    currentConfidence = data.confidence;
                    hasDetectedEmotion = true;
                    Debug.Log($"Emocao: {currentEmotion}, Confianca: {currentConfidence}");
                }
            }
            else
            {
                Debug.LogError("Erro ao acessar API: " + request.error);
            }
        }

        isProcessing = false;
    }

    void UpdateEmojiDisplay()
    {
        if (emojiImage == null) return;

        // Atualizar o emoji baseado na emocao atual
        switch (currentEmotion)
        {
            case "Happy":
                if (happyEmoji != null) emojiImage.sprite = happyEmoji;
                break;
            case "Sad":
                if (sadEmoji != null) emojiImage.sprite = sadEmoji;
                break;
            case "Angry":
                if (angryEmoji != null) emojiImage.sprite = angryEmoji;
                break;
            default:
                if (neutralEmoji != null) emojiImage.sprite = neutralEmoji;
                break;
        }

        // Atualizar texto central
        if (emotionCenterText != null)
        {
            if (isDetecting)
            {
                emotionCenterText.text = $"EMOTION: {currentEmotion.ToUpper()}";

                // emotionCenterText.text = $"EMOTION: {currentEmotion.ToUpper()} ({currentConfidence:P0})"; // com confianca
            }
            else
            {
                if (hasDetectedEmotion)
                {
                    emotionCenterText.text = $"EMOTION: {currentEmotion.ToUpper()}";
                }
                else
                {
                    emotionCenterText.text = "EMOTION: WAITING";
                }
            }
        }

        // Atualizar indicador de deteccao
        if (detectionIndicator != null)
        {
            detectionIndicator.SetActive(isDetecting);
        }

        // Ajustar transparancia baseado na confianca (hard coded sem transparencia, facilitar visualizacao)
        Color emojiColor = emojiImage.color;
        emojiColor.a = 1f;
        emojiImage.color = emojiColor;
    }
    void SetupCenterDisplay()
    {
        if (emotionCenterText != null)
        {
            emotionCenterText.text = "EMOTION: WAITING";
            //emotionCenterText.alignment = TextAlignmentOptions.Center;
            //emotionCenterText.fontSize = 24;
            //emotionCenterText.color = Color.white;
        }

        if (detectionIndicator != null)
        {
            detectionIndicator.SetActive(false);
        }
    }

    public string GetCurrentEmotion()
    {
        return currentEmotion;
    }

    public float GetCurrentConfidence()
    {
        return currentConfidence;
    }

    void OnDestroy()
    {
        if (webcamTexture != null && webcamTexture.isPlaying)
        {
            webcamTexture.Stop();
        }

        if (capturedTexture != null)
        {
            Destroy(capturedTexture);
        }
    }

    [System.Serializable]
    public class EmotionData
    {
        public string emotion;
        public float confidence;
    }
}