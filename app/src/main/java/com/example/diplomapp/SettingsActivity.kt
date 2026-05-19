package com.example.diplomapp

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.diplomapp.data.SettingsRepository
import com.example.diplomapp.service.CallMonitoringService
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton
import com.google.android.material.switchmaterial.SwitchMaterial

class SettingsActivity : AppCompatActivity() {
    private lateinit var settings: SettingsRepository
    private lateinit var tvAudioStatus: TextView
    private lateinit var tvPhoneStatus: TextView
    private lateinit var tvNotifyStatus: TextView
    private var pendingSwitchCallMonitoring: Boolean = false

    private val requestAudioLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        updatePermissionStatus()
        if (isGranted && pendingSwitchCallMonitoring) {
            checkPhonePermissionAndEnable()
        } else if (!isGranted) {
            findViewById<SwitchMaterial>(R.id.switchCallMonitoring).isChecked = false
            Toast.makeText(this, "Требуется разрешение для записи аудио", Toast.LENGTH_LONG).show()
        }
    }

    private val requestPhoneLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        updatePermissionStatus()
        if (isGranted && pendingSwitchCallMonitoring) {
            enableCallMonitoring(true)
        } else if (!isGranted) {
            findViewById<SwitchMaterial>(R.id.switchCallMonitoring).isChecked = false
            Toast.makeText(this, "Требуется разрешение на чтение состояния телефона", Toast.LENGTH_LONG).show()
        }
    }

    private val requestNotifyLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        updatePermissionStatus()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        settings = SettingsRepository(this)

        findViewById<MaterialToolbar>(R.id.toolbar).apply {
            setNavigationOnClickListener { finish() }
        }

        tvAudioStatus = findViewById(R.id.tvAudioStatus)
        tvPhoneStatus = findViewById(R.id.tvPhoneStatus)
        tvNotifyStatus = findViewById(R.id.tvNotifyStatus)

        findViewById<MaterialButton>(R.id.btnRequestAudio).setOnClickListener {
            requestAudioLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }

        findViewById<MaterialButton>(R.id.btnRequestPhone).setOnClickListener {
            requestPhoneLauncher.launch(Manifest.permission.READ_PHONE_STATE)
        }

        findViewById<MaterialButton>(R.id.btnRequestNotify).setOnClickListener {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                requestNotifyLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            } else {
                Toast.makeText(this, "Не требуется для этой версии Android", Toast.LENGTH_SHORT).show()
            }
        }

        findViewById<SwitchMaterial>(R.id.switchCallMonitoring).apply {
            isChecked = settings.isCallMonitoringEnabled
            setOnCheckedChangeListener { _, isChecked ->
                if (isChecked) {
                    checkAudioPermissionAndEnable()
                } else {
                    enableCallMonitoring(false)
                }
            }
        }

        findViewById<SwitchMaterial>(R.id.switchAutoCheck).apply {
            isChecked = settings.autoCheckOnCall
            setOnCheckedChangeListener { _, isChecked ->
                settings.autoCheckOnCall = isChecked
            }
        }

        updatePermissionStatus()
    }

    private fun updatePermissionStatus() {
        val audioGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        val phoneGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.READ_PHONE_STATE) == PackageManager.PERMISSION_GRANTED
        val notifyGranted = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED
        } else true

        tvAudioStatus.text = if (audioGranted) "✓ Предоставлено" else "✗ Не предоставлено"
        tvAudioStatus.setTextColor(ContextCompat.getColor(this, if (audioGranted) android.R.color.holo_green_dark else android.R.color.holo_red_dark))

        tvPhoneStatus.text = if (phoneGranted) "✓ Предоставлено" else "✗ Не предоставлено"
        tvPhoneStatus.setTextColor(ContextCompat.getColor(this, if (phoneGranted) android.R.color.holo_green_dark else android.R.color.holo_red_dark))

        tvNotifyStatus.text = if (notifyGranted) "✓ Предоставлено" else "✗ Не предоставлено"
        tvNotifyStatus.setTextColor(ContextCompat.getColor(this, if (notifyGranted) android.R.color.holo_green_dark else android.R.color.holo_red_dark))
    }

    private fun checkAudioPermissionAndEnable() {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED -> {
                checkPhonePermissionAndEnable()
            }
            else -> {
                pendingSwitchCallMonitoring = true
                requestAudioLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
    }

    private fun checkPhonePermissionAndEnable() {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.READ_PHONE_STATE) == PackageManager.PERMISSION_GRANTED -> {
                enableCallMonitoring(true)
            }
            else -> {
                pendingSwitchCallMonitoring = true
                requestPhoneLauncher.launch(Manifest.permission.READ_PHONE_STATE)
            }
        }
    }

    private fun enableCallMonitoring(enabled: Boolean) {
        settings.isCallMonitoringEnabled = enabled

        if (enabled) {
            val intent = Intent(this, CallMonitoringService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                startForegroundService(intent)
            } else {
                startService(intent)
            }
            Toast.makeText(this, "Мониторинг звонков включён", Toast.LENGTH_SHORT).show()
        } else {
            stopService(Intent(this, CallMonitoringService::class.java))
            Toast.makeText(this, "Мониторинг звонков выключен", Toast.LENGTH_SHORT).show()
        }
    }
}