package com.example.diplomapp

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.diplomapp.data.CheckHistory
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class CheckHistoryAdapter(
    private val onDeleteClick: (CheckHistory) -> Unit,
    private val onPlayClick: (CheckHistory) -> Unit
) : ListAdapter<CheckHistory, CheckHistoryAdapter.ViewHolder>(DiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_check_history, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class ViewHolder(itemView: android.view.View) : RecyclerView.ViewHolder(itemView) {
        private val tvFileName: android.widget.TextView = itemView.findViewById(R.id.tvFileName)
        private val tvResult: android.widget.TextView = itemView.findViewById(R.id.tvResult)
        private val tvConfidence: android.widget.TextView = itemView.findViewById(R.id.tvConfidence)
        private val tvDate: android.widget.TextView = itemView.findViewById(R.id.tvDate)
        private val tvModel: android.widget.TextView = itemView.findViewById(R.id.tvModel)
        private val indicator: android.view.View = itemView.findViewById(R.id.indicator)
        private val btnPlay: android.widget.ImageButton = itemView.findViewById(R.id.btnPlayHistory)

        fun bind(check: CheckHistory) {
            tvFileName.text = check.fileName
            tvResult.text = if (check.isVoiceFake) "Обнаружено: МОШЕННИЧЕСТВО" else "Голос настоящий"
            tvConfidence.text = "${(check.confidence * 100).toInt()}%"
            tvModel.text = check.modelUsed
            tvDate.text = SimpleDateFormat("dd.MM.yy HH:mm", Locale.getDefault())
                .format(Date(check.timestamp))

            val color = if (check.isVoiceFake) 
                android.R.color.holo_red_dark else android.R.color.holo_green_dark
            indicator.setBackgroundColor(ContextCompat.getColor(itemView.context, color))
            tvResult.setTextColor(ContextCompat.getColor(itemView.context, color))

            btnPlay.setOnClickListener { onPlayClick(check) }
            
            itemView.setOnLongClickListener {
                onDeleteClick(check)
                true
            }
        }
    }

    class DiffCallback : DiffUtil.ItemCallback<CheckHistory>() {
        override fun areItemsTheSame(oldItem: CheckHistory, newItem: CheckHistory) = 
            oldItem.id == newItem.id
        override fun areContentsTheSame(oldItem: CheckHistory, newItem: CheckHistory) = 
            oldItem == newItem
    }
}