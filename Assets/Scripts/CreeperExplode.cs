using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CreeperExplode : MonoBehaviour
{

    private AudioSource audioSource;

    void Start()
    {
        audioSource = GetComponent<AudioSource>();
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.name == "Player")
        {
            Debug.Log("Player-Creeper collision");

            // Animate explosion
            ParticleSystem exp = GetComponent<ParticleSystem>();
            exp.Play();
            audioSource.Play();

            // Destory Creeper after audio clip ended
            Invoke("OnCreeperExploded", audioSource.clip.length);
            // Destroy player
            Destroy(other.gameObject);
        }
    }

    void OnCreeperExploded()
    {
        Destroy(gameObject);
    }
}
