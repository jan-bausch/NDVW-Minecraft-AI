using UnityEngine;
using UnityEngine.UI;

public class UIManager : MonoBehaviour
{
    public Text numberOfCreepersText;
    public Text sizeOfWorldText;
    public Text frequencyOfPreciousBlocksText;
    public Text collectedPreciousBlocksText;
    public Text timerText;
    public Text agentPositionText;
    public Text enemiesInfoText;

    private bool isF3Pressed = false;
    private float updateInterval = 1;
    private float lastUpdateTime = 1.0f;

    private int timer = 100;
    private int collectedPreciousBlocks = 0;
    private int stopTimer = 0;

    void Start()
    {
        PlayerPrefs.SetInt("CollectedPreciousBlocks", collectedPreciousBlocks);
        PlayerPrefs.SetInt("Timer", timer);
        PlayerPrefs.SetInt("StopTimer", stopTimer);

        UpdateTimer(timer);
        UpdateCollectedPreciousBlocks(collectedPreciousBlocks);
        UpdateNumberOfCreepers(PlayerPrefs.GetInt("NumberOfCreepers"));
        UpdateFrequencyOfPreciousBlocks(PlayerPrefs.GetFloat("Frequency"));
        UpdateSizeOfWorld(PlayerPrefs.GetInt("SizeOfWorld"));
    }

    void Update()
    {
        isF3Pressed = Input.GetKey(KeyCode.F3);

        numberOfCreepersText.gameObject.SetActive(isF3Pressed);
        sizeOfWorldText.gameObject.SetActive(isF3Pressed);
        frequencyOfPreciousBlocksText.gameObject.SetActive(isF3Pressed);
        agentPositionText.gameObject.SetActive(isF3Pressed);
        enemiesInfoText.gameObject.SetActive(isF3Pressed);

        UpdateCollectedPreciousBlocks(PlayerPrefs.GetInt("CollectedPreciousBlocks"));

        if (Time.time - lastUpdateTime >= updateInterval)
        {
            lastUpdateTime = Time.time;
            UpdatePositions();
            timer = timer - 1;
            UpdateTimer(timer);
        }
    }

    void UpdatePositions()
    {
        Vector3 agentPosition = new Vector3(Random.Range(-10, 10), 0, Random.Range(-10, 10));
        agentPositionText.text = "Agent Position: " + Mathf.RoundToInt(agentPosition.x) + ", " + Mathf.RoundToInt(agentPosition.y) + ", " + Mathf.RoundToInt(agentPosition.z);

        int numberOfEnemies = Mathf.RoundToInt(PlayerPrefs.GetInt("NumberOfCreepers"));
        string enemiesInfo = "Enemies Info:\n";

        for (int i = 0; i < numberOfEnemies; i++)
        {
            Vector3 enemyPosition = new Vector3(Random.Range(-10, 10), 0, Random.Range(-10, 10));
            float distanceToEnemy = Vector3.Distance(agentPosition, enemyPosition);

            string enemyInfo = $"Enemy {i + 1}: {Mathf.RoundToInt(enemyPosition.x)}, {Mathf.RoundToInt(enemyPosition.y)}, {Mathf.RoundToInt(enemyPosition.z)} ({distanceToEnemy:F2} units)\n";
            enemiesInfo += enemyInfo;
        }

        enemiesInfoText.text = enemiesInfo;
    }

    public void UpdateNumberOfCreepers(float value)
    {
        numberOfCreepersText.text = "Number of Creepers: " + value.ToString();
    }

    public void UpdateSizeOfWorld(float value)
    {
        sizeOfWorldText.text = "Size of World: " + value.ToString();
    }

    public void UpdateFrequencyOfPreciousBlocks(float value)
    {
        frequencyOfPreciousBlocksText.text = "Frequency of Precious Blocks: " + value.ToString("F2");
    }

    public void UpdateCollectedPreciousBlocks(float value)
    {
        collectedPreciousBlocksText.text = "Collected Precious Blocks: " + value.ToString();
    }

    public void UpdateTimer(float value)
    {
        if (PlayerPrefs.GetInt("StopTimer") == 1)
        {
            Debug.Log("Game ended.");
        }
        else if (value < 0)
        {
            timerText.text = "Remaining Time: " + "0";
        }
        else 
        {
            timerText.text = "Remaining Time: " + Mathf.Round(value).ToString();
        }
    }
}
