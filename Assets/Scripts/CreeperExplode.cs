using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CreeperExplode : MonoBehaviour
{

    public AudioSource audioSource;

    void Start()
    {
        audioSource = GetComponent<AudioSource>();
    }

    void OnCollisionEnter(Collision other)
    {
        Debug.Log("Test");
        Destroy(gameObject);
        if (other.gameObject.name == "Player")
        {
            audioSource.Play();
            Destroy(gameObject);
        }
    }   
}
