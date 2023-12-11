using System;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace Environment
{
public class EnvironmentData
{
    public int[] gameState;
    public float rewardSignal;
}

public class EnvironmentDataSerializer
{

    public byte[] SerializeFrameInfo(int worldId, float rewardSignal, int[] gameState)
    {
        using (MemoryStream memoryStream = new MemoryStream())
        {
            using (BinaryWriter writer = new BinaryWriter(memoryStream))
            {
                writer.Write(worldId);
                writer.Write(rewardSignal);
                foreach (int pixel in gameState)
                {
                    writer.Write(pixel);
                }
            }

            return memoryStream.ToArray();
        }
    }

    public (int worldId, float rewardSignal, int[] gameState) DeserializeFrameInfo(byte[] data)
    {
        int worldId;
        float rewardSignal;
        int[] gameState;

        using (MemoryStream memoryStream = new MemoryStream(data))
        {
            using (BinaryReader reader = new BinaryReader(memoryStream))
            {
                worldId = reader.ReadInt32();
                rewardSignal = reader.ReadSingle();

                int remainingBytes = (int)(memoryStream.Length - memoryStream.Position);
                int pixelCount = remainingBytes / sizeof(int);

                gameState = new int[pixelCount];
                for (int i = 0; i < pixelCount; i++)
                {
                    gameState[i] = reader.ReadInt32();
                }
            }
        }

        return (worldId, rewardSignal, gameState);
    }
}

}