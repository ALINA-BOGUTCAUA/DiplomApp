package com.example.diplomapp.model

import android.content.Context
import android.util.Log
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okio.ByteString
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit

data class VoiceAnalysisResult(
    val isVoiceFake: Boolean,
    val confidence: Float,
    val modelResults: List<ModelPrediction>,
    val modelNames: List<String>,
    val modelUsed: String = "",
    val allModelResults: Map<String, Float> = emptyMap()
)

data class ModelPrediction(
    val modelName: String,
    val confidence: Float,
    val isFake: Boolean
)

class AudioClassifier(private val context: Context) {

    private val threshold = 0.5f
    private val SERVER_URL = "http://10.244.131.91:5000/predict" // Updated to your current PC IP
    
    // OkHttp client
    private val client = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    fun loadModels(): Boolean {
        Log.d("AudioClassifier", "Using remote server mode - no local models needed")
        return true
    }

    fun analyzeAll(audioData: FloatArray): VoiceAnalysisResult {
        return sendToServer(audioData, "full", "CNN")
    }

    fun analyzeWithModel(audioData: FloatArray, modelName: String): VoiceAnalysisResult {
        Log.d("AudioClassifier", "analyzeWithModel called with model: $modelName")
        return sendToServer(audioData, "model", modelName)
    }

    fun analyzeQuick(audioData: FloatArray): VoiceAnalysisResult {
        return sendToServer(audioData, "quick", "CNN")
    }

    private fun sendToServer(audioData: FloatArray, mode: String, modelName: String): VoiceAnalysisResult {
        Log.d("AudioClassifier", "sendToServer: mode=$mode, modelName=$modelName")
        return try {
            // Convert FloatArray to WAV file
            val audioFile = saveAudioToTempFile(audioData)
            
            // Create multipart request
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("mode", mode)
                .addFormDataPart("model_name", modelName)
                .addFormDataPart(
                    "audio",
                    "audio.wav",
                    RequestBody.create("audio/wav".toMediaType(), audioFile)
                )
                .build()

            val request = Request.Builder()
                .url(SERVER_URL)
                .post(requestBody)
                .build()

            val response = client.newCall(request).execute()
            val responseBody = response.body?.string()
            
            Log.d("AudioClassifier", "Server response: ${response.code}, body: $responseBody")

            // Clean up temp file
            audioFile.delete()

            if (response.isSuccessful && responseBody != null) {
                parseJsonResponse(responseBody)
            } else {
                Log.e("AudioClassifier", "Server error: ${response.code}")
                VoiceAnalysisResult(false, 0.5f, emptyList(), emptyList(), "", emptyMap())
            }
        } catch (e: Exception) {
            Log.e("AudioClassifier", "Server request failed: ${e.message}")
            VoiceAnalysisResult(false, 0.5f, emptyList(), emptyList(), "", emptyMap())
        }
    }

    private fun saveAudioToTempFile(audioData: FloatArray): File {
        val tempFile = File(context.cacheDir, "temp_audio_${System.currentTimeMillis()}.wav")
        
        // Convert FloatArray to WAV format
        val sampleRate = 16000
        val numChannels = 1
        val bitsPerSample = 16
        
        val dataSize = audioData.size * 2 // 16-bit = 2 bytes per sample
        val fileSize = 36 + dataSize
        
        val outputStream = FileOutputStream(tempFile)
        
        // WAV header
        outputStream.write("RIFF".toByteArray())
        writeInt(outputStream, fileSize) // File size - 8
        outputStream.write("WAVE".toByteArray())
        outputStream.write("fmt ".toByteArray())
        writeInt(outputStream, 16) // fmt chunk size
        writeShort(outputStream, 1) // Audio format (PCM)
        writeShort(outputStream, numChannels)
        writeInt(outputStream, sampleRate)
        writeInt(outputStream, sampleRate * numChannels * bitsPerSample / 8)
        writeShort(outputStream, (numChannels * bitsPerSample / 8))
        writeShort(outputStream, bitsPerSample)
        outputStream.write("data".toByteArray())
        writeInt(outputStream, dataSize)
        
        // Audio data
        for (sample in audioData) {
            val scaled = (sample * 32767f).coerceIn(-32768f, 32767f)
            writeShort(outputStream, scaled.toInt().toShort())
        }
        
        outputStream.close()
        return tempFile
    }

    private fun writeInt(outputStream: FileOutputStream, value: Int) {
        outputStream.write(value and 0xFF)
        outputStream.write((value shr 8) and 0xFF)
        outputStream.write((value shr 16) and 0xFF)
        outputStream.write((value shr 24) and 0xFF)
    }

    private fun writeShort(outputStream: FileOutputStream, value: Int) {
        outputStream.write(value and 0xFF)
        outputStream.write((value shr 8) and 0xFF)
    }

    private fun writeShort(outputStream: FileOutputStream, value: Short) {
        outputStream.write(value.toInt() and 0xFF)
        outputStream.write((value.toInt() shr 8) and 0xFF)
    }

    private fun parseJsonResponse(json: String): VoiceAnalysisResult {
        return try {
            val jsonObject = JSONObject(json)
            
            if (jsonObject.has("error")) {
                Log.e("AudioClassifier", "Server returned error: ${jsonObject.getString("error")}")
                return VoiceAnalysisResult(false, 0.5f, emptyList(), emptyList(), "", emptyMap())
            }
            
            val isFake = if (jsonObject.has("is_fake")) jsonObject.getBoolean("is_fake") else jsonObject.getBoolean("isVoiceFake")
            val confidence = if (jsonObject.has("confidence")) jsonObject.getDouble("confidence").toFloat() else jsonObject.getDouble("prob").toFloat()
            val modelUsed = jsonObject.optString("model_used", jsonObject.optString("modelUsed", "Unknown"))
            
            Log.d("AudioClassifier", "Parsed: isFake=$isFake, confidence=$confidence, modelUsed=$modelUsed")
            
            // Parse all_probs if available (full mode)
            val allProbsMap = mutableMapOf<String, Float>()
            if (jsonObject.has("all_probs")) {
                val allProbs = jsonObject.getJSONObject("all_probs")
                allProbs.keys().asSequence().forEach { key ->
                    allProbsMap[key] = allProbs.getDouble(key).toFloat()
                }
            }
            
            // Create predictions for each model
            val predictions = mutableListOf<ModelPrediction>()
            if (allProbsMap.isNotEmpty()) {
                allProbsMap.asSequence().forEach { (name, prob) ->
                    predictions.add(ModelPrediction(name, prob, prob > 0.5f))
                }
            } else {
                predictions.add(ModelPrediction(modelUsed, confidence, isFake))
            }
            
            VoiceAnalysisResult(
                isVoiceFake = isFake,
                confidence = confidence,
                modelResults = predictions,
                modelNames = predictions.map { it.modelName },
                modelUsed = modelUsed,
                allModelResults = allProbsMap
            )
        } catch (e: Exception) {
            Log.e("AudioClassifier", "JSON parse error: ${e.message}, json: $json")
            VoiceAnalysisResult(false, 0.5f, emptyList(), emptyList(), "", emptyMap())
        }
    }

    fun close() {
        Log.d("AudioClassifier", "Closing server client")
    }
}
