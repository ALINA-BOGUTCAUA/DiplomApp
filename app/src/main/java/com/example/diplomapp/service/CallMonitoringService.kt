package com.example.diplomapp.service

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import android.widget.Toast
import androidx.core.app.NotificationCompat
import com.example.diplomapp.FraudAlertActivity
import com.example.diplomapp.MainActivity
import com.example.diplomapp.R
import com.example.diplomapp.data.AppDatabase
import com.example.diplomapp.data.CheckHistory
import com.example.diplomapp.data.SettingsRepository
import com.example.diplomapp.model.AudioClassifier
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class CallMonitoringService : Service() {
    private var mediaRecorder: MediaRecorder? = null
    private var isRecording = false
    private var recordedFilePath: String? = null
    private lateinit var settings: SettingsRepository
    private val handler = Handler(Looper.getMainLooper())

    private var callCheckCount = 0
    private var currentCallId: String? = null

    private var audioClassifier: AudioClassifier? = null
    
    override fun onCreate() {
        super.onCreate()
        settings = SettingsRepository(this)
        audioClassifier = AudioClassifier(this)
        audioClassifier?.loadModels()
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, createNotification("Мониторинг звонков активен"))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        currentCallId = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        callCheckCount = 0

        if (settings.isCallMonitoringEnabled) {
            toast("Мониторинг звонка начат")
            startRecording(10)
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        audioClassifier?.close()
        stopRecording()
        handler.removeCallbacksAndMessages(null)
    }

    private fun startRecording(durationSeconds: Int) {
        if (isRecording) return
        if (callCheckCount >= 2) {
            toast("Проверка завершена. Звонок безопасен.")
            stopSelf()
            return
        }

        try {
            val dir = getExternalFilesDir(Environment.DIRECTORY_MUSIC)
            if (dir != null && !dir.exists()) dir.mkdirs()

            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            recordedFilePath = "${dir?.absolutePath}/call_${timestamp}.mp3"

            mediaRecorder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                MediaRecorder(this)
            } else {
                @Suppress("DEPRECATION")
                MediaRecorder()
            }.apply {
                setAudioSource(MediaRecorder.AudioSource.VOICE_COMMUNICATION)
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                setOutputFile(recordedFilePath)
                setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                prepare()
                start()
            }
            isRecording = true
            callCheckCount++

            toast("Запись ${callCheckCount}/2...")

            handler.postDelayed({
                stopRecordingAndCheck()
            }, durationSeconds * 1000L)

        } catch (e: Exception) {
            e.printStackTrace()
            toast("Ошибка записи: ${e.message}")
            stopSelf()
        }
    }

    private fun stopRecordingAndCheck() {
        if (!isRecording) return

        try {
            mediaRecorder?.apply {
                stop()
                release()
            }
            mediaRecorder = null
            isRecording = false

            recordedFilePath?.let { path ->
                val file = File(path)
                if (file.exists() && file.length() > 1000) {
                    toast("Проверка голоса...")
                    checkRecording(path)
                } else {
                    toast("Запись слишком короткая")
                    finishCall()
                }
            } ?: finishCall()

        } catch (e: Exception) {
            e.printStackTrace()
            finishCall()
        }
    }

    private fun checkRecording(filePath: String) {
        CoroutineScope(Dispatchers.Main).launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    performRealCheck(filePath)
                }

                val db = AppDatabase.getDatabase(this@CallMonitoringService)
                val check = CheckHistory(
                    fileName = File(filePath).name,
                    filePath = filePath,
                    modelUsed = result.modelUsed,
                    checkType = "call",
                    result = result.result,
                    confidence = result.confidence,
                    isVoiceFake = result.isVoiceFake
                )
                db.checkHistoryDao().insert(check)

                if (result.isVoiceFake) {
                    showFraudAlert(result.confidencePercent, result.modelUsed)
                } else {
                    showOkNotification(result.confidencePercent)
                    finishCall()
                }

            } catch (e: Exception) {
                e.printStackTrace()
                toast("Ошибка проверки")
                finishCall()
            }
        }
    }

    private fun performRealCheck(filePath: String): FraudCheckResult {
        return try {
            val audioData = loadAudioFromFile(filePath)
            
            val classifier = audioClassifier ?: throw Exception("Классификатор не инициализирован")
            
            val selectedModel = settings.defaultModel
            Log.d("CallMonitoring", "Using model: $selectedModel")
            val analysisResult = if (selectedModel.isEmpty() || selectedModel == "BASE_MODEL" || selectedModel == "CNN") {
                classifier.analyzeQuick(audioData)
            } else {
                classifier.analyzeWithModel(audioData, selectedModel)
            }
            
            FraudCheckResult(
                result = if (analysisResult.isVoiceFake) "Обнаружен синтезированный голос!" else "Голос настоящий",
                confidence = analysisResult.confidence,
                isVoiceFake = analysisResult.isVoiceFake,
                confidencePercent = (analysisResult.confidence * 100).toInt(),
                modelUsed = analysisResult.modelNames.joinToString(" + ")
            )
        } catch (e: Exception) {
            e.printStackTrace()
            FraudCheckResult(
                result = "Ошибка проверки: ${e.message}",
                confidence = 0f,
                isVoiceFake = false,
                confidencePercent = 0,
                modelUsed = "Ошибка"
            )
        }
    }
    
    private fun loadAudioFromFile(filePath: String): FloatArray {
        val uri = Uri.fromFile(File(filePath))
        val extractor = android.media.MediaExtractor()
        extractor.setDataSource(this, uri, null)
        
        var audioTrackIndex = -1
        var audioFormat: android.media.MediaFormat? = null
        
        for (i in 0 until extractor.trackCount) {
            val format = extractor.getTrackFormat(i)
            val mime = format.getString(android.media.MediaFormat.KEY_MIME)
            if (mime?.startsWith("audio/") == true) {
                audioTrackIndex = i
                audioFormat = format
                break
            }
        }
        
        if (audioTrackIndex == -1 || audioFormat == null) {
            extractor.release()
            throw Exception("Аудио дорожка не найдена")
        }
        
        extractor.selectTrack(audioTrackIndex)
        
        val sampleRate = audioFormat.getInteger(android.media.MediaFormat.KEY_SAMPLE_RATE)
        val durationUs = audioFormat.getLong(android.media.MediaFormat.KEY_DURATION)
        val expectedSamples = (durationUs / 1_000_000.0 * sampleRate).toInt()
        val samples = minOf(expectedSamples, 48000)
        
        val buffer = java.nio.ByteBuffer.allocate(samples * 2)
        val bufferInfo = android.media.MediaCodec.BufferInfo()
        
        val decoder = android.media.MediaCodec.createDecoderByType(
            audioFormat.getString(android.media.MediaFormat.KEY_MIME) ?: "audio/mp4a-latm"
        )
        decoder.configure(audioFormat, null, null, 0)
        decoder.start()
        
        var inputDone = false
        var outputDone = false
        val pcmData = mutableListOf<Float>()
        
        while (!outputDone) {
            if (!inputDone) {
                val inputBufferIndex = decoder.dequeueInputBuffer(10000)
                if (inputBufferIndex >= 0) {
                    val inputBuffer = decoder.getInputBuffer(inputBufferIndex)
                    val sampleSize = extractor.readSampleData(inputBuffer!!, 0)
                    if (sampleSize < 0) {
                        decoder.queueInputBuffer(inputBufferIndex, 0, 0, 0, android.media.MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputDone = true
                    } else {
                        val presentationTimeUs = extractor.sampleTime
                        decoder.queueInputBuffer(inputBufferIndex, 0, sampleSize, presentationTimeUs, 0)
                        extractor.advance()
                    }
                }
            }
            
            val outputBufferIndex = decoder.dequeueOutputBuffer(bufferInfo, 10000)
            if (outputBufferIndex >= 0) {
                val outputBuffer = decoder.getOutputBuffer(outputBufferIndex)
                if (bufferInfo.size > 0 && outputBuffer != null) {
                    outputBuffer.position(bufferInfo.offset)
                    val shortBuffer = outputBuffer.asShortBuffer()
                    while (shortBuffer.hasRemaining() && pcmData.size < samples) {
                        pcmData.add(shortBuffer.get().toFloat() / 32768f)
                    }
                }
                decoder.releaseOutputBuffer(outputBufferIndex, false)
                if (bufferInfo.flags and android.media.MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                    outputDone = true
                }
            }
        }
        
        decoder.stop()
        decoder.release()
        extractor.release()
        
        val result = FloatArray(48000)
        val copySize = minOf(pcmData.size, 48000)
        for (i in 0 until copySize) {
            result[i] = pcmData[i]
        }
        
        return result
    }

    private fun showFraudAlert(confidence: Int, modelUsed: String) {
        stopRecording()

        val intent = Intent(this, FraudAlertActivity::class.java).apply {
            putExtra(FraudAlertActivity.EXTRA_CONFIDENCE, confidence)
            putExtra(FraudAlertActivity.EXTRA_MODEL, modelUsed)
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        startActivity(intent)

        val notificationIntent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, notificationIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val alertNotification = NotificationCompat.Builder(this, ALERT_CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_warning)
            .setContentTitle("⚠️ ВНИМАНИЕ! МОШЕННИК")
            .setContentText("Обнаружен синтезированный голос (${confidence}%)")
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .setContentIntent(pendingIntent)
            .build()

        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(ALERT_NOTIFICATION_ID + callCheckCount, alertNotification)

        toast("⚠️ ВНИМАНИЕ! Возможен мошенник!")
        
        finishCall()
    }

    private fun showOkNotification(confidence: Int) {
        val notificationIntent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, notificationIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val okNotification = NotificationCompat.Builder(this, ALERT_CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_verified)
            .setContentTitle("✅ Проверка пройдена")
            .setContentText("Голос настоящий (${confidence}%)")
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(true)
            .setContentIntent(pendingIntent)
            .build()

        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(OK_NOTIFICATION_ID, okNotification)
    }

    private fun finishCall() {
        toast("Проверка завершена. Звонок безопасен.")
        stopSelf()
    }

    private fun stopRecording() {
        try {
            mediaRecorder?.apply {
                stop()
                release()
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        mediaRecorder = null
        isRecording = false
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(CHANNEL_ID, "Мониторинг звонков", NotificationManager.IMPORTANCE_LOW).apply {
                description = "Мониторинг звонков"
            }
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)

            val alertChannel = NotificationChannel(ALERT_CHANNEL_ID, "Оповещения", NotificationManager.IMPORTANCE_HIGH).apply {
                description = "Оповещения о проверке"
            }
            notificationManager.createNotificationChannel(alertChannel)
        }
    }

    private fun createNotification(text: String) = NotificationCompat.Builder(this, CHANNEL_ID)
        .setSmallIcon(R.drawable.ic_mic)
        .setContentText(text)
        .setPriority(NotificationCompat.PRIORITY_LOW)
        .build()

    private fun toast(message: String) {
        Handler(Looper.getMainLooper()).post {
            Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
        }
    }

    data class FraudCheckResult(
        val result: String,
        val confidence: Float,
        val isVoiceFake: Boolean,
        val confidencePercent: Int,
        val modelUsed: String
    )

    companion object {
        private const val CHANNEL_ID = "call_monitoring_channel"
        private const val ALERT_CHANNEL_ID = "alert_channel"
        private const val NOTIFICATION_ID = 1
        private const val ALERT_NOTIFICATION_ID = 100
        private const val OK_NOTIFICATION_ID = 200
    }
}