# pip install openai python-dotenv pydub streamlit

import os
import re
import zipfile
from dotenv import load_dotenv
import openai
from datetime import datetime
from pydub import AudioSegment
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# st.subheader("DEBUG: Inhalt von st.secrets")
# st.write(st.secrets.to_dict())

# -- Konfiguration und Initialisierung --

# Lade Umgebungsvariablen für den lokalen Betrieb.
# In Streamlit Cloud werden stattdessen die "Secrets" verwendet.
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    # Die Authenticator-Konfiguration wird auch aus den Secrets geladen
    # Manuelle Konvertierung von st.secrets in ein veränderliches dict, um TypeErrors zu vermeiden.
    # --- CHANGE START ---
    # The 'preauthorized' key is deprecated and has been removed.
    config = {
        'credentials': {
            'usernames': dict(st.secrets['credentials']['usernames'])
        },
        'cookie': dict(st.secrets['cookie']),
    }
    # --- CHANGE END ---
except (KeyError, FileNotFoundError):
    # Lokaler Fallback
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise KeyError()
        with open('config.yaml') as file:
            config = yaml.load(file, Loader=SafeLoader)
    except Exception:
        st.error("Fehler: Kritische Konfigurationen (API-Key oder Auth-Config) fehlen.")
        st.stop()

# Initialisiere den OpenAI-Client mit dem geladenen API-Schlüssel.
# Dieses Client-Objekt wird für alle Anfragen an die OpenAI-API verwendet.
client = openai.OpenAI(api_key=api_key)

# Definiere globale Konstanten für Dateipfade, um den Code lesbarer und wartbarer zu machen.
ZIP_FILE = "chat.zip"
EXTRACT_DIR = "extracted_chat"
CHAT_FILE_NAME = "_chat.txt"


def unzip_chat(zip_path, extract_to):
    """
    Entpackt die angegebene ZIP-Datei in ein Zielverzeichnis und sucht nach der Chat-Textdatei.

    Args:
        zip_path (str): Der Pfad zur ZIP-Datei (z.B. "chat.zip").
        extract_to (str): Das Verzeichnis, in das die Inhalte entpackt werden sollen.

    Returns:
        str: Der vollständige Pfad zur gefundenen _chat.txt-Datei.

    Raises:
        FileNotFoundError: Wenn die ZIP-Datei oder die _chat.txt-Datei nach dem Entpacken nicht gefunden wird.
    """
    # Stelle sicher, dass die ZIP-Datei existiert, bevor versucht wird, sie zu öffnen.
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Die Datei '{zip_path}' wurde nicht im aktuellen Verzeichnis gefunden.")

    # Erstelle das Zielverzeichnis, falls es noch nicht existiert.
    os.makedirs(extract_to, exist_ok=True)

    # Öffne die ZIP-Datei und entpacke alle ihre Inhalte in das Zielverzeichnis.
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Suche nach der Chat-Textdatei (_chat.txt) im entpackten Verzeichnis.
    chat_txt_path = os.path.join(extract_to, CHAT_FILE_NAME)
    if not os.path.exists(chat_txt_path):
        # Versuche, die Datei zu finden, falls sie in einem Unterverzeichnis liegt
        for root, _, files in os.walk(extract_to):
            if CHAT_FILE_NAME in files:
                chat_txt_path = os.path.join(root, CHAT_FILE_NAME)
                return chat_txt_path
        raise FileNotFoundError(f"Konnte '{CHAT_FILE_NAME}' im entpackten Verzeichnis nicht finden.")
    
    return chat_txt_path


def transcribe_audio(audio_path):
    """
    Transkribiert eine einzelne Audiodatei mit der OpenAI Whisper API.
    Konvertiert .opus-Dateien automatisch in .mp3, da .opus nicht unterstützt wird.

    Args:
        audio_path (str): Der Pfad zur Audiodatei (z.B. .ogg, .opus).

    Returns:
        str: Der transkribierte Text der Audionachricht.
    """
    file_extension = os.path.splitext(audio_path)[1].lower()
    path_to_transcribe = audio_path
    temp_file_to_clean = None

    # Konvertiere .opus zu .mp3, falls nötig
    if file_extension == ".opus":
        try:
            # Entferne explizites format="opus", damit ffmpeg das Format auto-erkennen kann.
            # Das ist robuster und vermeidet den "Unknown input format"-Fehler.
            audio = AudioSegment.from_file(audio_path)
            temp_mp3_path = os.path.splitext(audio_path)[0] + ".mp3"
            audio.export(temp_mp3_path, format="mp3")
            path_to_transcribe = temp_mp3_path
            temp_file_to_clean = temp_mp3_path
        except Exception as e:
            print(f"Fehler bei der Konvertierung von {os.path.basename(audio_path)}: {e}")
            print("Stellen Sie sicher, dass 'ffmpeg' installiert ist (z.B. mit 'brew install ffmpeg').")
            return "[Fehler bei der Konvertierung]"

    try:
        with open(path_to_transcribe, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcription.text
    except Exception as e:
        print(f"Fehler bei der Transkription von {os.path.basename(audio_path)}: {e}")
        return "[Fehler bei der Transkription]"
    finally:
        # Räume die temporäre Datei auf, falls eine erstellt wurde
        if temp_file_to_clean and os.path.exists(temp_file_to_clean):
            os.remove(temp_file_to_clean)


def build_full_transcript(chat_txt_path, base_dir, start_dt=None):
    """
    Liest die Chat-Datei blockweise (jede Nachricht inkl. Folgezeilen), ersetzt Audio-Verweise durch Transkriptionen und erstellt ein vollständiges Transkript.
    Optional: Nur Nachrichten ab start_dt werden übernommen.

    Args:
        chat_txt_path (str): Der Pfad zur _chat.txt-Datei.
        base_dir (str): Das Basisverzeichnis, in dem die Mediendateien zu finden sind.
        start_dt (datetime.datetime, optional): Startdatum als datetime-Objekt.

    Returns:
        str: Ein einziger String, der den gesamten Chatverlauf enthält.
    """
    # Muster für Zeilen, die mit einem Datum beginnen.
    # WICHTIG: Erlaubt ein optionales unsichtbares Steuerzeichen (\u200e), das WhatsApp manchmal am Zeilenanfang einfügt.
    date_line_pattern = re.compile(r"^\u200e?\[(\d{2}\.\d{2}\.\d{2,4}), ")
    
    # Flexibleres Muster, das verschiedene Audio-Formate (.ogg, .opus) und Export-Stile erkennt.
    # Es sucht nach <Anhang: ...> oder <Medien ausgeschlossen> (...)
    audio_line_pattern = re.compile(
        r"(\[.*?\] .*?:).*?(?:<Anhang: (.*?\.(?:opus|ogg))>|<Medien ausgeschlossen> \(Datei angehängt: (.*?\.(?:opus|ogg))\))"
    )

    # Startdatum als datetime-Objekt parsen, falls angegeben
    full_transcript_lines = []
    
    with open(chat_txt_path, 'r', encoding='utf-8') as file:
        current_block = []
        current_block_date = None
        for line in file:
            date_match = date_line_pattern.match(line)
            if date_match:
                # Blockwechsel: alten Block ggf. übernehmen
                if current_block and (not start_dt or (current_block_date and current_block_date >= start_dt)):
                    full_transcript_lines.extend(current_block)
                # Neuen Block starten
                current_block = [line]
                date_str = date_match.group(1)
                try:
                    if len(date_str.split(".")[-1]) == 2:
                        current_block_date = datetime.strptime(date_str, "%d.%m.%y")
                    else:
                        current_block_date = datetime.strptime(date_str, "%d.%m.%Y")
                except ValueError:
                    current_block_date = None
            else:
                # Zeile gehört zum aktuellen Block
                current_block.append(line)
        # Letzten Block nach der Schleife prüfen und ggf. hinzufügen
        if current_block and (not start_dt or (current_block_date and current_block_date >= start_dt)):
            full_transcript_lines.extend(current_block)

    # Audiozeilen im gefilterten Transkript ersetzen
    result_lines = []
    audio_transcriptions = {} # Cache
    for line in full_transcript_lines:
        match = audio_line_pattern.search(line)
        if match:
            metadata_part = match.group(1)
            # Der Dateiname ist in Gruppe 2 oder 3, je nachdem, welcher Teil des Musters traf
            audio_filename = match.group(2) or match.group(3)

            # Sicherheitshalber prüfen, ob ein Dateiname gefunden wurde
            if not audio_filename:
                result_lines.append(line)
                continue

            # Transkription durchführen oder aus dem Cache laden
            if audio_filename in audio_transcriptions:
                transcribed_text = audio_transcriptions[audio_filename]
            else:
                audio_path = os.path.join(base_dir, audio_filename)
                transcribed_text = transcribe_audio(audio_path)
                audio_transcriptions[audio_filename] = transcribed_text
            
            new_line = f"{metadata_part} [AUDIO] {transcribed_text}\n"
            result_lines.append(new_line)
        else:
            result_lines.append(line)

    return "".join(result_lines)


def summarize_text(text_to_summarize):
    """
    Sendet den aufbereiteten Chat-Text an die OpenAI API zur Zusammenfassung.

    Args:
        text_to_summarize (str): Der vollständige, aufbereitete Chat-Verlauf.

    Returns:
        str: Die von der KI generierte Zusammenfassung.
    """
    # Der System-Prompt gibt der KI die Anweisung, wie sie sich verhalten soll.
    # Er definiert die Rolle, die Aufgabe und das gewünschte Ausgabeformat.
    system_prompt = """Du bist ein Experte darin, unstrukturierte WhatsApp-Chatverläufe zu analysieren. Deine Aufgabe ist es, eine klare, prägnante und nützliche Zusammenfassung zu erstellen. Fasse die wichtigsten Punkte, Entscheidungen, offenen Fragen und die allgemeine Stimmung in Stichpunkten zusammen. Beginne direkt mit der Zusammenfassung und antworte auf Deutsch."""

    try:
        # Sende die Anfrage an die Chat Completions API.
        response = client.chat.completions.create(
            model="gpt-4o",  # Verwende das leistungsstarke und effiziente gpt-4o Modell.
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_to_summarize}
            ]
        )
        # Extrahiere die Textantwort aus dem API-Response-Objekt.
        summary = response.choices[0].message.content
        return summary
    except openai.APIError as e:
        st.error(f"Fehler bei der Erstellung der Zusammenfassung: {e}")
        return "Die Zusammenfassung konnte aufgrund eines API-Fehlers nicht erstellt werden."


def run_summarizer_app():
    """Die eigentliche Logik der Anwendung, die nach dem Login ausgeführt wird."""
    st.title("WhatsApp Chat Summarizer")
    st.info("Dieses Tool analysiert einen WhatsApp-Chat-Export, transkribiert Audionachrichten und erstellt eine Zusammenfassung mit GPT-4o.")
    start_date = st.date_input("Wählen Sie das Startdatum für die Analyse aus.", value=None)
    uploaded_file = st.file_uploader("Laden Sie Ihre 'chat.zip'-Exportdatei hier hoch", type=['zip'])
    if st.button("Zusammenfassung starten"):
        if not start_date:
            st.error("Fehler: Bitte wählen Sie ein Startdatum aus.")
        elif not uploaded_file:
            st.error("Fehler: Bitte laden Sie Ihre 'chat.zip'-Datei hoch.")
        else:
            start_dt = datetime.combine(start_date, datetime.min.time())
            with open(ZIP_FILE, "wb") as f:
                f.write(uploaded_file.getvalue())
            try:
                with st.spinner("Verarbeite Chat... Dies kann einen Moment dauern."):
                    chat_txt_path = unzip_chat(ZIP_FILE, EXTRACT_DIR)
                    media_base_dir = os.path.dirname(chat_txt_path)
                    full_transcript_text = build_full_transcript(chat_txt_path, media_base_dir, start_dt)
                    final_summary = summarize_text(full_transcript_text)
                st.subheader("KI-generierte Zusammenfassung")
                st.markdown(final_summary)
                with st.expander("Vollständiges Transkript anzeigen (gekürzt)"):
                    st.code(full_transcript_text[:3000] + "...", language=None)
            except Exception as e:
                st.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
            finally:
                if os.path.exists(ZIP_FILE):
                    os.remove(ZIP_FILE)


def main():
    """Hauptfunktion, die die Streamlit-Anwendung startet."""

    # Setze das Seitenlayout auf "wide" für mehr Platz.
    st.set_page_config(page_title="Chat Summarizer", layout="wide")
    
    # --- CHANGE START ---
    # The 'preauthorized' parameter is deprecated and must be removed from the call.
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    # --- CHANGE END ---

    # Starte den Authentifizierungsprozess.
    authenticator.login()

    if st.session_state["authentication_status"]:
        # Notwendig, um die Konfiguration zu speichern, falls das Passwort gehasht wurde
        # If 'preauthorized' existed in the original yaml, it will not be written back,
        # effectively cleaning the file on the first successful login.
        with open('config.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        authenticator.logout()
        st.write(f'Willkommen, *{st.session_state["name"]}*!')
        run_summarizer_app()
    elif st.session_state.get("authentication_status") is False:
        st.error('Benutzername/Passwort ist inkorrekt')
    elif st.session_state.get("authentication_status") is None:
        st.warning('Bitte geben Sie Ihren Benutzernamen und Ihr Passwort ein')


if __name__ == "__main__":
    main()