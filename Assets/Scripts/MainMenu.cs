using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class MainMenuScript : MonoBehaviour
{
    public Slider numberOfCreepersSlider;
    public Text numberOfCreepersText;

    public Slider sizeOfWorldSlider;
    public Text sizeOfWorldText;

    public Slider frequencySlider;
    public Text frequencyText;

    public Button startGameButton;

    public GameObject gameScene;

    public UIManager uiManager;

    void Start()
    {
        numberOfCreepersSlider.minValue = 1;
        numberOfCreepersSlider.maxValue = 4;
        numberOfCreepersSlider.value = 2;

        sizeOfWorldSlider.minValue = 20;
        sizeOfWorldSlider.maxValue = 100;
        sizeOfWorldSlider.value = 20;

        frequencySlider.minValue = 0;
        frequencySlider.maxValue = 1;
        frequencySlider.value = 0.5f;

        uiManager.UpdateNumberOfCreepers(numberOfCreepersSlider.value);
        uiManager.UpdateSizeOfWorld(sizeOfWorldSlider.value);
        uiManager.UpdateFrequencyOfPreciousBlocks(frequencySlider.value);

        UpdateSliderTexts();

        numberOfCreepersSlider.onValueChanged.AddListener(OnNumberOfCreepersValueChanged);
        sizeOfWorldSlider.onValueChanged.AddListener(OnSizeOfWorldValueChanged);
        frequencySlider.onValueChanged.AddListener(OnFrequencyValueChanged);

        startGameButton.onClick.AddListener(StartGame);

    }
    public void StartGame()
    {
        Debug.Log("Starting game...");
        SceneManager.LoadScene(1);
    }

    public void UpdateSliderTexts()
    {
        numberOfCreepersText.text = "" + Mathf.RoundToInt(numberOfCreepersSlider.value);
        sizeOfWorldText.text = "" + Mathf.RoundToInt(sizeOfWorldSlider.value);
        frequencyText.text = "" + frequencySlider.value.ToString("F2");
    }

    public void OnNumberOfCreepersValueChanged(float value)
    {
        numberOfCreepersSlider.value = Mathf.Clamp(Mathf.RoundToInt(value), numberOfCreepersSlider.minValue, numberOfCreepersSlider.maxValue);
        UpdateSliderTexts();
        uiManager.UpdateNumberOfCreepers(numberOfCreepersSlider.value);
        PlayerPrefs.SetInt("NumberOfCreepers", Mathf.RoundToInt(numberOfCreepersSlider.value));
    }

    public void OnSizeOfWorldValueChanged(float value)
    {
        sizeOfWorldSlider.value = Mathf.Clamp(Mathf.RoundToInt(value), sizeOfWorldSlider.minValue, sizeOfWorldSlider.maxValue);
        UpdateSliderTexts(); 
        uiManager.UpdateSizeOfWorld(sizeOfWorldSlider.value);
        PlayerPrefs.SetInt("SizeOfWorld", Mathf.RoundToInt(sizeOfWorldSlider.value));
    }

    public void OnFrequencyValueChanged(float value)
    {
        frequencySlider.value = Mathf.Clamp(Mathf.Round(value / 0.05f) * 0.05f, frequencySlider.minValue, frequencySlider.maxValue);
        UpdateSliderTexts();
        uiManager.UpdateFrequencyOfPreciousBlocks(frequencySlider.value);
        PlayerPrefs.SetFloat("Frequency", frequencySlider.value);
    }
}
