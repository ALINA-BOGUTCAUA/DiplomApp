package com.example.diplomapp

import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.WindowManager
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity

class FraudAlertActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON or
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON)

        val confidence = intent.getIntExtra(EXTRA_CONFIDENCE, 0)
        val modelUsed = intent.getStringExtra(EXTRA_MODEL) ?: "Базовая нейросеть"

        showFraudAlert(confidence, modelUsed)
    }

    private fun showFraudAlert(confidence: Int, modelUsed: String) {
        AlertDialog.Builder(this)
            .setTitle("ВНИМАНИЕ: Возможно мошенник!")
            .setMessage("Обнаружен синтезированный голос\n\nВероятность: $confidence%\nМодель: $modelUsed")
            .setIcon(R.drawable.ic_warning)
            .setPositiveButton("ОК") { _, _ ->
                dismissNotification()
                finish()
            }
            .setCancelable(false)
            .show()
    }

    private fun dismissNotification() {
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.cancel(FRAUD_NOTIFICATION_ID)
    }

    companion object {
        const val EXTRA_CONFIDENCE = "confidence"
        const val EXTRA_MODEL = "model"
        const val FRAUD_NOTIFICATION_ID = 2
    }
}