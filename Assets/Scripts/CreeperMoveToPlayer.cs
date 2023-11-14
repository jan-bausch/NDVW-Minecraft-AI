using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CreeperLookAtPlayer : MonoBehaviour
{

    public Transform target;
    public float moveSpeed = 2.5f;
    public float detectionRadius = 5.0f;

    void Update()
    {
        // Check if player is close
        if ((target.transform.position - transform.position).magnitude < detectionRadius)
        {
            // Target player
            transform.LookAt(new Vector3(target.position.x, transform.position.y, target.position.z));

            // Move forward
            transform.position += transform.forward * Time.deltaTime * moveSpeed;
        }
    }
}
