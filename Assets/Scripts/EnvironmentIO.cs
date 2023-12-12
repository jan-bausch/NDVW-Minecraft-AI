using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Player;
using System;
using Creeper;

namespace Environment
{
    public class EnvironmentIO : MonoBehaviour
    {
        public bool jumping = false;
        public bool goingLeft = false;
        public bool goingRight = false;
        public bool goingForward = false;
        public bool goingBackward = false;  
        public bool placingBlock = false;
        public bool breakingBlock = false;  
        public bool lookingLeft = false;
        public bool lookingRight = false;
        public bool lookingUp = false;
        public bool lookingDown = false;

        private RenderTexture renderTexture;
        
        public void ResetAllInputs()
        {
            jumping = false;
            goingLeft = false;
            goingRight = false;
            goingForward = false;
            goingBackward = false;  
            placingBlock = false;
            breakingBlock = false;  
            lookingLeft = false;
            lookingRight = false;
            lookingUp = false;
            lookingDown = false;
        }

        public void ToggleInput(int id)
        {
            if (id == 0) jumping = true;
            if (id == 1) goingLeft = true;
            if (id == 2) goingRight = true;
            if (id == 3) goingForward = true;
            if (id == 4) goingBackward = true;  
            if (id == 5) placingBlock = true;
            if (id == 6) breakingBlock = true;  
            if (id == 7) lookingLeft = true;
            if (id == 8) lookingRight = true;
            if (id == 9) lookingUp = true;
            if (id == 10) lookingDown = true;
        }

        public int[] GetGameState()
        {
            if (renderTexture == null) return null;
        
            RenderTexture.active = renderTexture;

            // Create a texture2D and read pixels
            Texture2D tex = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
            tex.Apply();

            Color[] pixels = tex.GetPixels();
            int[] channelsflattened = new int[pixels.Length*3];

            for (int i = 0; i < pixels.Length; i++)
            {
                channelsflattened[i] = (int)(pixels[i].r * 255f);
            }
            for (int i = pixels.Length; i < pixels.Length*2; i++)
            {
                channelsflattened[i] = (int)(pixels[i-pixels.Length].g * 255f);
            }
            for (int i = pixels.Length*2; i < pixels.Length*3; i++)
            {
                channelsflattened[i] = (int)(pixels[i-(pixels.Length*2)].b * 255f);
            }

            RenderTexture.active = null;

            int additionalStates = 2; 
            int[] gameState = new int[pixels.Length*3 + additionalStates];
            for (int i = additionalStates; i < gameState.Length; i++)
            {
                gameState[i] = channelsflattened[i-additionalStates];
            }

            Transform playerTransform = transform.Find("Player");
            
            BlockInteraction bi = playerTransform.gameObject.GetComponent<BlockInteraction>();

            var (invSolid, invPrecious) = bi.GetInv();
            gameState[0] = invSolid;
            gameState[1] = invPrecious;
            
            return gameState;
        }

        public float GetReward()
        {
            Transform playerTransform = transform.Find("Player");
            Transform creeperTransform = transform.Find("Creeper");

            if (playerTransform.position.y < 0.0f)
            {
                return -1.0f;
            }
            float distance = (1.5f * Math.Max(Math.Min(Vector3.Distance(playerTransform.position, creeperTransform.position) / 20.0f, 1.0f), 0.0f)) -1.0f;
            
            BlockInteraction bi = playerTransform.gameObject.GetComponent<BlockInteraction>();
            var (invSolid, invPrecious) = bi.GetInv();
            float precious = Math.Max(Math.Min((float) invPrecious / 10.0f, 1.0f), 0.0f);

            float reward = distance + 0.5f * precious;
            return reward;
        }

        public void EnableRemoteControlled()
        {
            Transform playerTransform = transform.Find("Player");
            
            BlockInteraction bi = playerTransform.gameObject.GetComponent<BlockInteraction>();
            bi.remoteControlled = true;

            PlayerMovement pm = playerTransform.gameObject.GetComponent<PlayerMovement>();
            pm.remoteControlled = true;

            Transform creeperTransform = transform.Find("Creeper");
            CreeperMovement cm = creeperTransform.gameObject.GetComponent<CreeperMovement>();
            cm.remoteControlled = true;

            CreeperExplode ce = creeperTransform.gameObject.GetComponent<CreeperExplode>();
            ce.remoteControlled = true;

            Transform camTransform = transform.Find("CameraHolder").Find("Camera");
           
            PlayerCam pc = camTransform.gameObject.GetComponent<PlayerCam>();
            pc.remoteControlled = true;

            Camera cameraComponent = camTransform.gameObject.GetComponent<Camera>();
            renderTexture = new RenderTexture(64, 64, 24);
            renderTexture.name = "EnvironmentRenderTexture";
            cameraComponent.targetTexture = renderTexture;
        }

        public void MoveUpdate(float delta)
        {
            Transform playerTransform = transform.Find("Player");
            
            BlockInteraction bi = playerTransform.gameObject.GetComponent<BlockInteraction>();
            bi.placingBlock = placingBlock;
            bi.breakingBlock = breakingBlock;
            bi.MoveUpdate(delta);
            
            PlayerMovement pm = playerTransform.gameObject.GetComponent<PlayerMovement>();
            pm.jumping = jumping;
            pm.goingLeft = goingLeft;
            pm.goingRight = goingRight;
            pm.goingForward = goingForward;
            pm.goingBackward = goingBackward;
            pm.MoveUpdate(delta);

            Transform creeperTransform = transform.Find("Creeper");
            
            CreeperMovement cm = creeperTransform.gameObject.GetComponent<CreeperMovement>();
            cm.MoveUpdate(delta);

            Transform camTransform = transform.Find("CameraHolder").Find("Camera");
           
            PlayerCam pc = camTransform.gameObject.GetComponent<PlayerCam>();
            pc.lookingLeft = lookingLeft;
            pc.lookingRight = lookingRight;
            pc.lookingUp = lookingUp;
            pc.lookingDown = lookingDown;
            pc.MoveUpdate(delta);
        }
    }
}