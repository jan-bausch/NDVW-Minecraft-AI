using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Creeper 
{
    public class CreeperExplode : MonoBehaviour
    {

        private AudioSource audioSource;

        void Start()
        {
            audioSource = GetComponent<AudioSource>();
        }

        void OnTriggerStay(Collider other)
        {
            //Debug.Log(other.gameObject);
            if (other.gameObject.name == "Player")
            {
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
}