package com.example.diplomapp

import com.example.diplomapp.model.VoiceAnalysisResult
import com.example.diplomapp.model.ModelPrediction

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaMetadataRetriever
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.OpenableColumns
import android.view.View
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.diplomapp.data.AppDatabase
import com.example.diplomapp.data.CheckHistory
import com.example.diplomapp.data.SettingsRepository
import com.example.diplomapp.model.AudioClassifier
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.HashMap

class MainActivity : AppCompatActivity() {

    private val pickAudio = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            audioUri = it
            etAudioFile.setText(getFileName(it))
            tvAudioInfo.text = getAudioInfo(it)
            Toast.makeText(this, "Аудиофайл выбран: ${getFileName(it)}", Toast.LENGTH_SHORT).show()
        }
    }

    private val requestPermissions = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
        val allGranted = permissions.all { it.value }
        if (!allGranted) {
            Toast.makeText(this, "Разрешения отклонены. Запись недоступна.", Toast.LENGTH_LONG).show()
        }
    }

    private var audioUri: Uri? = null
    private var mediaRecorder: MediaRecorder? = null
    private var audioFilePath: String? = null

    private lateinit var etAudioFile: TextInputEditText
    private lateinit var etAudioUrl: TextInputEditText
    private lateinit var btnRecord: MaterialButton
    private lateinit var btnStop: MaterialButton
    private lateinit var btnPlay: MaterialButton
    private lateinit var btnRecognize: MaterialButton
    private lateinit var tvAudioInfo: TextView
    private lateinit var tvResult: TextView
    private lateinit var tvConfidence: TextView
    private lateinit var tvAllModelsResult: TextView
    private lateinit var cardResult: View
    private lateinit var cardProgress: View
    private lateinit var rgCheckType: RadioGroup
    private lateinit var spinnerModel: android.widget.AutoCompleteTextView
    private lateinit var cardModel: View

    private var mediaPlayer: android.media.MediaPlayer? = null
    
    private lateinit var settings: SettingsRepository
    private lateinit var db: AppDatabase
    private lateinit var audioClassifier: AudioClassifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        settings = SettingsRepository(this)
        db = AppDatabase.getDatabase(this)
        
        audioClassifier = AudioClassifier(this)
        audioClassifier.loadModels()

        initViews()
        setupListeners()
        checkAndRequestPermissions()

        findViewById<View>(R.id.btnHistory).setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }

        findViewById<View>(R.id.btnSettings).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
    }

    override fun onResume() {
        super.onResume()
        checkAndRequestPermissions()
    }

    private fun initViews() {
        etAudioFile = findViewById(R.id.etAudioFile)
        etAudioUrl = findViewById(R.id.etAudioUrl)
        btnRecord = findViewById(R.id.btnRecord)
        btnStop = findViewById(R.id.btnStop)
        btnPlay = findViewById(R.id.btnPlay)
        btnRecognize = findViewById(R.id.btnRecognize)
        tvAudioInfo = findViewById(R.id.tvAudioInfo)
        tvResult = findViewById(R.id.tvResult)
        tvConfidence = findViewById(R.id.tvConfidence)
        tvAllModelsResult = findViewById(R.id.tvAllModelsResult)
        cardResult = findViewById(R.id.cardResult)
        cardProgress = findViewById(R.id.cardProgress)
        rgCheckType = findViewById(R.id.rgCheckType)
        spinnerModel = findViewById(R.id.spinnerModel)
        cardModel = findViewById(R.id.cardModel)

        btnStop.isEnabled = false

        val models = listOf("CNN", "LSTM", "RCNN", "RF", "Wav2Vec")
        spinnerModel.setText(settings.defaultModel.ifEmpty { models[0] })
        cardModel.visibility = View.GONE
        
        val modelArray = android.widget.ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, models)
        spinnerModel.setAdapter(modelArray)
        
        spinnerModel.setOnItemClickListener { _, _, position, _ ->
            settings.defaultModel = models[position]
        }
        
        spinnerModel.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) {
                val text = spinnerModel.text.toString()
                if (text in models) {
                    settings.defaultModel = text
                }
            }
        }
    }

    private fun setupListeners() {
        etAudioFile.setOnClickListener { pickAudio.launch("audio/*") }

        findViewById<View>(R.id.etAudioUrl).setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) loadAudioFromUrl()
        }

        btnRecord.setOnClickListener { startRecording() }
        btnStop.setOnClickListener { stopRecording() }
        btnPlay.setOnClickListener { playAudio() }
        btnRecognize.setOnClickListener { startCheck() }

        rgCheckType.setOnCheckedChangeListener { _, checkedId ->
            val isModel = checkedId == R.id.rbModel
            val description = when (checkedId) {
                R.id.rbFast -> "Быстрая проверка: CNN (через сервер)"
                R.id.rbFull -> "Долгая проверка: все 5 моделей (через сервер)"
                R.id.rbModel -> "Выберите модель"
                else -> ""
            }
            cardModel.visibility = if (isModel) View.VISIBLE else View.GONE
            findViewById<TextView>(R.id.tvCheckDescription).text = description
        }
    }

    private fun checkAndRequestPermissions() {
        val permissions = mutableListOf<String>()
        
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.RECORD_AUDIO)
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_PHONE_STATE) != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.READ_PHONE_STATE)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                permissions.add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
        
        if (permissions.isNotEmpty()) {
            requestPermissions.launch(permissions.toTypedArray())
        }
    }

    private fun loadAudioFromUrl() {
        val url = etAudioUrl.text.toString()
        if (url.isEmpty()) return
        try {
            audioUri = Uri.parse(url)
            etAudioFile.setText(url)
            tvAudioInfo.text = getAudioInfo(audioUri!!)
            Toast.makeText(this, "Аудиофайл загружен по ссылке", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Неверный URL", Toast.LENGTH_SHORT).show()
        }
    }

    private fun getFileName(uri: Uri): String {
        var result = "unknown"
        try {
            if (uri.scheme == "file") {
                result = uri.lastPathSegment ?: File(uri.path!!).name
            } else {
                contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                    if (cursor.moveToFirst()) {
                        val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                        if (index >= 0) result = cursor.getString(index)
                    }
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return result
    }

    private fun getAudioInfo(uri: Uri): String {
        var info = "Информация недоступна"
        try {
            val mmr = MediaMetadataRetriever()
            if (uri.scheme == "file") {
                val file = File(uri.path!!)
                if (file.exists()) mmr.setDataSource(file.absolutePath)
            } else if (uri.scheme == "content") {
                mmr.setDataSource(this, uri)
            } else {
                mmr.setDataSource(uri.toString(), HashMap())
            }

            val mime = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_MIMETYPE)
            val durationStr = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val durationMs = durationStr?.toLongOrNull() ?: 0
            val minutes = String.format("%02d", (durationMs / 1000) / 60)
            val seconds = String.format("%02d", (durationMs / 1000) % 60)
            val fileName = getFileName(uri)

            info = buildString {
                appendLine("Файл: $fileName")
                appendLine("Длительность: $minutes:$seconds")
                if (mime != null) appendLine("Формат: $mime")
            }.trimEnd()
            mmr.release()
        } catch (e: Exception) {
            e.printStackTrace()
            val fileName = getFileName(uri)
            info = "Файл: $fileName"
        }
        return info
    }

    private fun startRecording() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Нет разрешения на запись аудио", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val dir = getExternalFilesDir(Environment.DIRECTORY_MUSIC)
            if (dir != null && !dir.exists()) dir.mkdirs()

            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            audioFilePath = "${getExternalFilesDir(Environment.DIRECTORY_MUSIC)?.absolutePath}/voice_${timestamp}.mp3"

            mediaRecorder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                MediaRecorder(this)
            } else {
                @Suppress("DEPRECATION")
                MediaRecorder()
            }.apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                setOutputFile(audioFilePath)
                setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                prepare()
                start()
            }

            btnRecord.isEnabled = false
            btnStop.isEnabled = true
            Toast.makeText(this, "Начата запись...", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Ошибка при записи аудио", Toast.LENGTH_SHORT).show()
        }
    }

    private fun stopRecording() {
        mediaRecorder?.apply {
            try {
                stop()
                release()
                mediaRecorder = null

                audioUri = Uri.fromFile(File(audioFilePath!!))

                val displayName = File(audioFilePath!!).name
                tvAudioInfo.text = getAudioInfo(audioUri!!)
                etAudioFile.setText(displayName)

                btnRecord.isEnabled = true
                btnStop.isEnabled = false
                Toast.makeText(this@MainActivity, "Запись остановлена", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private fun playAudio() {
        val filePath = audioFilePath
        if (filePath != null) {
            playFile(File(filePath))
            return
        }
        
        audioUri?.let { uri ->
            try {
                val path = uri.path
                if (path != null) {
                    playFile(File(path))
                    return
                }
            } catch (e: Exception) {}
        }
        Toast.makeText(this, "Сначала выберите или запишите аудиофайл", Toast.LENGTH_SHORT).show()
    }
    
    private fun playFile(file: File) {
        if (!file.exists()) {
            Toast.makeText(this, "Файл не найден", Toast.LENGTH_SHORT).show()
            return
        }
        try {
            mediaPlayer?.release()
            mediaPlayer = android.media.MediaPlayer().apply {
                setDataSource(file.absolutePath)
                prepare()
                start()
                setOnCompletionListener {
                    release()
                    mediaPlayer = null
                }
            }
            Toast.makeText(this, "Воспроизведение...", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Ошибка: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startCheck() {
        if (audioUri == null) {
            Toast.makeText(this, "Сначала выберите аудиофайл", Toast.LENGTH_SHORT).show()
            return
        }

        val checkedId = rgCheckType.checkedRadioButtonId
        val isFastCheck = checkedId == R.id.rbFast
        val isFullCheck = checkedId == R.id.rbFull
        val isModelCheck = checkedId == R.id.rbModel
        val model = spinnerModel.text.toString()
        val checkType = when (checkedId) {
            R.id.rbFast -> "fast"
            R.id.rbFull -> "full"
            R.id.rbModel -> "model"
            else -> "fast"
        }

        cardProgress.visibility = View.VISIBLE
        cardResult.visibility = View.GONE
        btnRecognize.isEnabled = false

        lifecycleScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    checkVoice(isFastCheck, isFullCheck, isModelCheck, model)
                }

                cardProgress.visibility = View.GONE
                cardResult.visibility = View.VISIBLE
                btnRecognize.isEnabled = true

                var resultText = result.result
                
                tvResult.text = resultText
                tvResult.setTextColor(
                    if (result.isVoiceFake) ContextCompat.getColor(this@MainActivity, android.R.color.holo_red_dark)
                    else ContextCompat.getColor(this@MainActivity, android.R.color.holo_green_dark)
                )
                
                // Show all models results if available
                if (result.allModelResults.isNotEmpty()) {
                    val modelsText = StringBuilder()
                    result.allModelResults.forEach { (name, prob) ->
                        val label = if (prob > 0.5f) "🟥 SPOOF" else "🟩 REAL"
                        modelsText.appendLine("$name: ${String.format("%.3f", prob)} - $label")
                    }
                    tvAllModelsResult.text = modelsText.toString().trimStart()
                    tvAllModelsResult.visibility = View.VISIBLE
                } else {
                    tvAllModelsResult.visibility = View.GONE
                }
                
                tvConfidence.text = result.modelUsed
                tvConfidence.setTextColor(ContextCompat.getColor(this@MainActivity, android.R.color.darker_gray))

                tvAudioInfo.text = "Проверка завершена"

                val fileName = getFileName(audioUri!!)
                val history = CheckHistory(
                    fileName = fileName,
                    filePath = audioUri.toString(),
                    modelUsed = result.modelUsed,
                    checkType = checkType,
                    result = result.result,
                    confidence = result.confidence,
                    isVoiceFake = result.isVoiceFake
                )
                db.checkHistoryDao().insert(history)

            } catch (e: Exception) {
                cardProgress.visibility = View.GONE
                cardResult.visibility = View.VISIBLE
                btnRecognize.isEnabled = true
                Toast.makeText(this@MainActivity, "Ошибка: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

private fun checkVoice(isFastCheck: Boolean, isFullCheck: Boolean, isModelCheck: Boolean, selectedModel: String): CheckResult {
        try {
            val audioData = loadAudioFromUri(audioUri!!)
            
            val result = when {
                isFastCheck -> audioClassifier.analyzeQuick(audioData)
                isModelCheck -> audioClassifier.analyzeWithModel(audioData, selectedModel)
                else -> audioClassifier.analyzeAll(audioData)
            }
            
            val resultText = if (result.isVoiceFake) {
                "Да"
            } else {
                "Нет"
            }
            
            val modelNames: String = result.modelNames.joinToString(" + ").ifEmpty { result.modelUsed }
            
                return CheckResult(
                    result = resultText,
                    confidence = result.confidence,
                    isVoiceFake = result.isVoiceFake,
                    confidencePercent = (result.confidence * 100).toInt(),
                    modelUsed = modelNames,
                    allResults = null,
                    allModelResults = result.allModelResults
                )
        } catch (e: Exception) {
            e.printStackTrace()
return CheckResult(
                result = "Ошибка: ${e.message}",
                confidence = 0f,
                isVoiceFake = false,
                confidencePercent = 0,
                modelUsed = "",
                allResults = null,
                allModelResults = emptyMap()
            )
        }
    }
    
    private fun loadAudioFromUri(uri: Uri): FloatArray {
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

    data class CheckResult(
        val result: String,
        val confidence: Float,
        val isVoiceFake: Boolean,
        val confidencePercent: Int,
        val modelUsed: String,
        val allResults: List<ModelResult>?,
        val allModelResults: Map<String, Float> = emptyMap()
    )

    data class ModelResult(
        val model: String,
        val confidence: Int,
        val isVoiceFake: Boolean
    )
}