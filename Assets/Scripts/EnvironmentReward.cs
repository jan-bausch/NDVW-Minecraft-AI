using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

namespace Environment {
public class EnvironmentReward : MonoBehaviour
{
    public float GetReward() {
        Transform playerTransform = transform.Find("Player");
        Vector3 pos = playerTransform.position;
        //float reward = Math.Max(Math.Min(1.0f, (pos.y - 9.0f) / (20.0f - 9.0f)), -1.0f);
        Vector3 center = transform.position + new Vector3(10.0f, 10.0f, 10.0f);
        double _reward = Math.Pow(center.x - pos.x, 2) + Math.Pow(center.y - pos.y, 2) + Math.Pow(center.y - pos.y, 2);
        float reward = (float) (-1.0 * Math.Max(Math.Min(_reward / 300.0, 1.0), -1.0));
        return reward;
    }
}
}
