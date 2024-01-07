using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class EndScene : MonoBehaviour
{
    public Text EndSceneText;
    public Text EndSceneText_shadow;
    public Button mainMenuButton;
    public Button exitButton;

    void Start()
    {
        Cursor.visible = true;
        Cursor.lockState = CursorLockMode.None;
        EndSceneText.text = PlayerPrefs.GetString("EndSceneText", "Ye");
        EndSceneText_shadow.text = EndSceneText.text;

        mainMenuButton.onClick.AddListener(LoadMainMenu);
        exitButton.onClick.AddListener(ExitGame);

    }

    public void LoadMainMenu()
    {
        SceneManager.LoadScene(0);
    }


    public void ExitGame()
    {
        Application.Quit();
    }

    // void Update()
    // {
    //     float mouseX = Input.GetAxis("Mouse X");
    //     float mouseY = Input.GetAxis("Mouse Y");
    // }
}