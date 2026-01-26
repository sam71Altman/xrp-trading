# XRP/USDT Telegram Signals Bot - Design Guidelines

## Project Context
This is a **backend Telegram bot** that sends trading signals. There is **no mobile app or custom frontend UI**. The entire user experience occurs through Telegram messages and bot commands.

## Brand Identity
**Purpose**: Deliver clear, actionable trading signals to Arabic-speaking traders with minimal noise and maximum clarity.

**Personality**: Professional, trustworthy, and direct. No unnecessary jargon. Signals should feel like receiving advice from a knowledgeable, cautious trader—not a gambling machine.

**Memorable Element**: Crisp, emoji-enhanced message formatting that makes entry/exit data scannable at a glance.

---

## Message Design System

### Visual Hierarchy in Messages
All Telegram messages must follow this structure:

**Signal Type Header** (emoji + bold text)  
🟢 **إشارة شراء** (BUY)  
🔴 **إغلاق المركز** (EXIT)  
ℹ️ **حالة البوت** (STATUS)

**Key Data** (labeled, each on new line)  
📊 **الإطار الزمني**: 1m  
💰 **سعر الدخول**: 2.1450  
🎯 **جني الأرباح**: 2.1536 (+0.40%)  
🛑 **وقف الخسارة**: 2.1386 (-0.30%)

**Reason/Context** (concise explanation)  
📈 **السبب**: اختراق أعلى سعر للـ5 شموع السابقة

### Typography Rules
- Use **bold** for labels and critical numbers (entry, TP, SL)
- Use regular text for explanations
- Use emojis sparingly but consistently:
  - 🟢 BUY signal
  - 🔴 EXIT signal
  - 📊 Timeframe
  - 💰 Entry price
  - 🎯 Take profit
  - 🛑 Stop loss
  - 📈/📉 Reason/trend
  - ⚠️ Warnings
  - ✅ Confirmations

### Message Templates

**BUY Signal:**
```
🟢 **إشارة شراء - XRPUSDT**

📊 **الإطار الزمني**: 1m
💰 **سعر الدخول**: 2.1450
🎯 **جني الأرباح**: 2.1536 (+0.40%)
🛑 **وقف الخسارة**: 2.1386 (-0.30%)

📈 **السبب**: EMA20 > EMA50 + اختراق أعلى سعر
```

**EXIT Signal:**
```
🔴 **إغلاق المركز - XRPUSDT**

💰 **سعر الخروج**: 2.1540
📊 **الربح/الخسارة**: +0.42%
✅ **السبب**: وصول الهدف
```

**Command Responses:**
- `/start` → "✅ البوت يعمل | الحالة: نشط"
- `/status` → Show position state (open/closed), entry if open, last candle close
- `/settf 5m` → "✅ تم تغيير الإطار الزمني إلى 5m"
- `/on` → "✅ تم تفعيل إرسال الإشارات"
- `/off` → "⏸️ تم إيقاف إرسال الإشارات مؤقتاً"

**Error Messages:**
- Network errors: "⚠️ خطأ في الاتصال. المحاولة مجدداً..."
- Invalid command: "❌ أمر غير صحيح. استخدم /start للمساعدة"

---

## Bot Behavior & UX Rules

### Anti-Spam Logic
- Maximum 1 message per minute (even if multiple signals trigger)
- No duplicate BUY signals while position is open
- No duplicate EXIT signals when no position exists
- If cooldown is active, queue the most important signal only

### Error Handling
- Network/API failures should NOT stop the bot
- Log errors silently, retry after 60s
- If repeated failures (5+ consecutive), send ONE alert to chat: "⚠️ مشكلة في الاتصال بـBinance"

### State Transparency
- `/status` command should always work and show:
  - Current mode (ON/OFF)
  - Position state (open/closed)
  - If open: entry price, current PnL
  - Last candle close price
  - Active timeframe

---

## Assets
*No custom assets required.* This is a text-only Telegram bot. All visual elements are emojis and text formatting.

---

## Technical Constraints
- Telegram message length limit: 4096 characters (keep messages under 500 chars)
- Arabic text direction: Right-to-left (RTL) supported natively by Telegram
- Emoji rendering: Standard Unicode emojis work across all Telegram clients