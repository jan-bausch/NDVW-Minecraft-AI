using System.Collections.Generic;
using UnityEngine;

public class InGameConsole : MonoBehaviour
{
    private List<string> logs = new List<string>(); // Store logs in a list
    private Vector2 scrollPosition;

    void Start()
    {
        // Set the screen resolution (change values as needed)
        Screen.SetResolution(1280, 720, false); // Width, Height, Fullscreen (false for windowed)
        Cursor.visible = true;
    }

    void OnEnable()
    {
        Application.logMessageReceived += HandleLog; // Subscribe to log events
    }

    void OnDisable()
    {
        Application.logMessageReceived -= HandleLog; // Unsubscribe from log events
    }

    void HandleLog(string logString, string stackTrace, LogType type)
    {
        // Add logs to the list
        logs.Add(logString);

        // Limit the number of logs to display (adjust as needed)
        if (logs.Count > 50)
        {
            logs.RemoveAt(0);
        }
    }

    void OnGUI()
    {
        // Draw the console window
        GUILayout.BeginArea(new Rect(10, 10, 400, 300));
        GUILayout.Label("In-Game Console");

        // Create a scroll view for logs
        scrollPosition = GUILayout.BeginScrollView(scrollPosition);

        // Display logs in the console window
        foreach (string log in logs)
        {
            GUILayout.Label(log);
        }

        GUILayout.EndScrollView();
        GUILayout.EndArea();
    }
}
