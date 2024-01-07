using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class EndScene : MonoBehaviour
{
    public Text EndSceneText;
    public Text EndSceneText_shadow;

    void Start()
    {
        EndSceneText.text = PlayerPrefs.GetString("EndSceneText", "Ye");
        EndSceneText_shadow.text = EndSceneText.text;
    }

}