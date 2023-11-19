using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

namespace Environment {
public class EnvironmentReward : MonoBehaviour
{
    public float GetReward() {
        Transform playerTransform = transform.Find("Player");
        return Math.Max(Math.Min(20.0f, playerTransform.position.y / 20.0f), 0.0f);
    }
}
}
