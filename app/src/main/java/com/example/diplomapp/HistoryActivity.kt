package com.example.diplomapp

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.diplomapp.data.AppDatabase
import com.example.diplomapp.data.CheckHistory
import com.example.diplomapp.data.CheckHistoryRepository
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import java.io.File

class HistoryActivity : AppCompatActivity() {
    private lateinit var repository: CheckHistoryRepository
    private lateinit var adapter: CheckHistoryAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_history)
        
        setSupportActionBar(findViewById(R.id.toolbar))
        supportActionBar?.setDisplayHomeAsUpEnabled(true)

        val db = AppDatabase.getDatabase(this)
        repository = CheckHistoryRepository(db.checkHistoryDao())

        val recyclerView = findViewById<androidx.recyclerview.widget.RecyclerView>(R.id.recyclerView)
        adapter = CheckHistoryAdapter(
            onDeleteClick = { check -> showDeleteDialog(check) },
            onPlayClick = { check -> playAudio(check) }
        )
        recyclerView.adapter = adapter
        recyclerView.layoutManager = LinearLayoutManager(this)

        lifecycleScope.launch {
            repository.allChecks.collectLatest { checks ->
                adapter.submitList(checks)
                findViewById<View>(R.id.emptyView).visibility = 
                    if (checks.isEmpty()) View.VISIBLE else View.GONE
            }
        }
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_history, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_clear -> {
                showClearAllDialog()
                true
            }
            android.R.id.home -> {
                finish()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun playAudio(check: CheckHistory) {
        try {
            val filePath = check.filePath
            if (filePath.startsWith("/")) {
                val file = File(filePath)
                if (file.exists()) {
                    val uri = Uri.fromFile(file)
                    val intent = Intent(Intent.ACTION_VIEW).apply {
                        setDataAndType(uri, "audio/*")
                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }
                    startActivity(intent)
                } else {
                    Toast.makeText(this, "Файл не найден", Toast.LENGTH_SHORT).show()
                }
            } else {
                val uri = Uri.parse(filePath)
                val intent = Intent(Intent.ACTION_VIEW).apply {
                    setDataAndType(uri, "audio/*")
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                startActivity(intent)
            }
        } catch (e: Exception) {
            Toast.makeText(this, "Не удалось воспроизвести", Toast.LENGTH_SHORT).show()
        }
    }

    private fun showDeleteDialog(check: CheckHistory) {
        MaterialAlertDialogBuilder(this)
            .setTitle("Удалить запись")
            .setMessage("Удалить эту проверку из истории?")
            .setPositiveButton("Удалить") { _, _ ->
                lifecycleScope.launch { repository.deleteCheck(check.id) }
            }
            .setNegativeButton("Отмена", null)
            .show()
    }

    private fun showClearAllDialog() {
        MaterialAlertDialogBuilder(this)
            .setTitle("Очистить историю")
            .setMessage("Удалить все записи из истории?")
            .setPositiveButton("Очистить") { _, _ ->
                lifecycleScope.launch { repository.clearHistory() }
            }
            .setNegativeButton("Отмена", null)
            .show()
    }
}