using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Player;

namespace Creeper 
{
    public class CreeperExplode : MonoBehaviour
    {

        private AudioSource audioSource;

        public bool remoteControlled;

        void Start()
        {
            audioSource = GetComponent<AudioSource>();
        }

        void OnTriggerStay(Collider other)
        {
            //Debug.Log(other.gameObject);
            if (!remoteControlled && other.gameObject.name == "Player")
            {
                // Animate explosion
                ParticleSystem exp = GetComponent<ParticleSystem>();
                exp.Play();
                audioSource.Play();

                // Destory Creeper after audio clip ended
                Invoke("OnCreeperExploded", audioSource.clip.length);
                // Destroy player
                Destroy(other.gameObject);
                // Stop timer
                PlayerPrefs.SetInt("StopTimer", 1);
            }
            if (remoteControlled && other.gameObject.name == "Player")
            {
                PlayerMovement pm = other.gameObject.GetComponent<PlayerMovement>();
                pm.dead = true;
            }
        }

        void OnCreeperExploded()
        {
            Destroy(gameObject);
        }
    }
}
