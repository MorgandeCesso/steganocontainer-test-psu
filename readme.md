# Запуск

# Генерация документа с стегоконтейнером внутри изображения
```bash
python main.py embed --cover-png cover.png --message "Секретный блок отчёта A3. Доступ: роль Инженер." --password "StrongPass123" --out-docx report.docx
```

# Извлечение стегоконтейнера из документа
```bash
python main.py extract --in-docx report.docx --password "StrongPass123"
```