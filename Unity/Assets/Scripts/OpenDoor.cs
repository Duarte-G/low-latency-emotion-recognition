using UnityEngine;

public class EmotionDoorController : MonoBehaviour
{
    [Header("Emotion Detection")]
    public EmotionDetectionSystem emotionSystem;

    [Header("Emotion Required")]
    public string requiredEmotion = "Happy";
    public float minimumConfidence = 0.7f;

    [Header("Door Settings")]
    public Transform door;
    public float openAngle = 90f;
    public float openSpeed = 2f;

    private Quaternion closedRotation;
    private Quaternion openRotation;
    private bool shouldOpen = false;

    void Start()
    {
        if (door == null)
        {
            door = transform;
        }

        closedRotation = door.rotation;
        openRotation = Quaternion.Euler(
            door.eulerAngles.x,
            door.eulerAngles.y + openAngle,
            door.eulerAngles.z
        );
    }

    void Update()
    {
        if (emotionSystem == null) return;

        string currentEmotion = emotionSystem.GetCurrentEmotion();
        float currentConfidence = emotionSystem.GetCurrentConfidence();

        if (currentEmotion == requiredEmotion && currentConfidence >= minimumConfidence)
        {
            shouldOpen = true;
        }
        else
        {
            shouldOpen = false;
        }

        if (shouldOpen)
        {
            door.rotation = Quaternion.Lerp(
                door.rotation,
                openRotation,
                Time.deltaTime * openSpeed
            );
        }
        else
        {
            door.rotation = Quaternion.Lerp(
                door.rotation,
                closedRotation,
                Time.deltaTime * openSpeed
            );
        }
    }
}