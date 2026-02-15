# Design Document: Nagrik-Sahayak (AI Citizen Helper)

## Overview

Nagrik-Sahayak is a voice-first, AI-powered mobile web application that provides rural Indian citizens with accessible information about government schemes. The system architecture follows a microservices pattern with clear separation between the frontend interface, backend API services, AI/ML processing pipeline, and data storage layers.

The core workflow follows this path:
1. User speaks in native language → Voice captured via mobile interface
2. Speech-to-Text (Whisper/Bhashini) → Transcription
3. Translation (Bhashini API) → English query
4. RAG Pipeline (LangChain + Vector DB) → Relevant scheme documents retrieved
5. LLM Reasoning (Gemini/GPT-4) → Contextual response generation
6. Translation (Bhashini API) → Response in user's language
7. Text-to-Speech → Audio playback

The system prioritizes low latency, offline capability, and mobile-first design to serve users with limited connectivity and literacy.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Mobile Web Frontend                     │
│                    (React/Streamlit - PWA)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Voice Input  │  │  Language    │  │   Display    │           │
│  │  Component   │  │  Selector    │  │  Component   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTPS/REST API
┌────────────────────────────┴────────────────────────────────────┐
│                      Backend API Layer                          │
│                   (FastAPI - Python)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Session    │  │    Query     │  │   Scheme     │           │
│  │  Management  │  │  Processing  │  │   Admin      │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                    AI/ML Processing Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Bhashini   │  │  RAG Engine  │  │     LLM      │           │
│  │  (STT/TTS/   │  │  (LangChain) │  │   (Gemini/   │           │
│  │  Translation)│  │              │  │    GPT-4)    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                        Data Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Vector DB  │  │    Redis     │  │  PostgreSQL  │           │
│  │  (Pinecone/  │  │   (Session   │  │  (Analytics/ │           │
│  │   Chroma)    │  │    Cache)    │  │   Metadata)  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Frontend:**
- React.js with TypeScript for UI components
- Progressive Web App (PWA) for offline capability
- Web Speech API for browser-native voice input
- Tailwind CSS for responsive mobile-first design
- Service Workers for caching and offline support

**Backend:**
- FastAPI (Python 3.11+) for REST API
- Pydantic for data validation
- Redis for session management and caching
- PostgreSQL for analytics and metadata storage

**AI/ML Services:**
- Bhashini API for Indian language STT/TTS/Translation
- OpenAI Whisper (fallback for STT)
- LangChain for RAG orchestration
- Pinecone or ChromaDB for vector storage
- Google Gemini or OpenAI GPT-4 for LLM reasoning
- Sentence Transformers for embedding generation

**Infrastructure:**
- Docker containers for service deployment
- Nginx for reverse proxy and load balancing
- AWS/GCP for cloud hosting
- CloudFront/CDN for static asset delivery

## Components and Interfaces

### 1. Voice Interface Component

**Responsibilities:**
- Capture audio input from user's microphone
- Provide visual feedback during recording
- Handle automatic silence detection
- Manage audio playback for responses

**Interfaces:**

```typescript
interface VoiceInputComponent {
  startRecording(): Promise<void>;
  stopRecording(): Promise<AudioBlob>;
  onSilenceDetected(callback: () => void): void;
  getRecordingState(): RecordingState;
}

enum RecordingState {
  IDLE,
  RECORDING,
  PROCESSING
}

interface AudioBlob {
  data: Blob;
  duration: number;
  format: string;
}
```

**Implementation Notes:**
- Uses Web Speech API or MediaRecorder API
- Implements Voice Activity Detection (VAD) for 2-second silence threshold
- Supports both manual stop and automatic stop
- Provides visual waveform or pulsing animation during recording

### 2. Language Selector Component

**Responsibilities:**
- Display available languages on first launch
- Allow language switching during session
- Persist language preference in browser storage
- Update UI text based on selected language

**Interfaces:**

```typescript
interface LanguageSelectorComponent {
  displayLanguageOptions(): void;
  selectLanguage(languageCode: string): void;
  getCurrentLanguage(): Language;
  updateUILanguage(languageCode: string): void;
}

interface Language {
  code: string;  // ISO 639-1 code
  name: string;  // Native name (e.g., "हिंदी", "தமிழ்")
  bhashiniCode: string;  // Bhashini API language code
}
```

### 3. Session Management Service

**Responsibilities:**
- Create and maintain user sessions
- Store conversation context and history
- Handle session expiration (30 minutes inactivity)
- Clear personal data on session end

**Interfaces:**

```python
class SessionManager:
    def create_session(self, language_code: str) -> Session
    def get_session(self, session_id: str) -> Optional[Session]
    def update_context(self, session_id: str, context: ConversationContext) -> None
    def clear_session(self, session_id: str) -> None
    def check_expiration(self, session_id: str) -> bool

@dataclass
class Session:
    session_id: str
    language_code: str
    created_at: datetime
    last_activity: datetime
    conversation_context: ConversationContext

@dataclass
class ConversationContext:
    messages: List[Message]
    mentioned_schemes: List[str]
    user_info: Dict[str, Any]  # Eligibility data
```

### 4. Speech-to-Text Service

**Responsibilities:**
- Transcribe audio to text using Bhashini API
- Support 10 Indian languages + English
- Handle transcription errors gracefully
- Return confidence scores

**Interfaces:**

```python
class SpeechToTextService:
    def transcribe(self, audio: bytes, language_code: str) -> TranscriptionResult
    def is_language_supported(self, language_code: str) -> bool

@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    language_detected: str
    error: Optional[str]
```

### 5. Translation Service

**Responsibilities:**
- Translate regional languages to English for processing
- Translate English responses back to user's language
- Preserve semantic meaning and context
- Handle translation failures

**Interfaces:**

```python
class TranslationService:
    def translate_to_english(self, text: str, source_language: str) -> TranslationResult
    def translate_from_english(self, text: str, target_language: str) -> TranslationResult

@dataclass
class TranslationResult:
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    error: Optional[str]
```

### 6. RAG Engine

**Responsibilities:**
- Generate embeddings for user queries
- Search vector database for relevant scheme documents
- Retrieve top-k most similar document chunks
- Pass retrieved context to LLM

**Interfaces:**

```python
class RAGEngine:
    def generate_embedding(self, text: str) -> np.ndarray
    def search_schemes(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SchemeChunk]
    def retrieve_context(self, query: str) -> RetrievalResult

@dataclass
class SchemeChunk:
    chunk_id: str
    scheme_name: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float

@dataclass
class RetrievalResult:
    chunks: List[SchemeChunk]
    query: str
    total_results: int
```

### 7. LLM Service

**Responsibilities:**
- Generate contextual responses using retrieved documents
- Avoid hallucination by grounding in retrieved context
- Structure responses in simple, conversational language
- Include scheme names, benefits, and eligibility criteria

**Interfaces:**

```python
class LLMService:
    def generate_response(self, query: str, context: List[SchemeChunk], 
                         conversation_history: List[Message]) -> LLMResponse
    def validate_response(self, response: str, context: List[SchemeChunk]) -> bool

@dataclass
class LLMResponse:
    response_text: str
    schemes_mentioned: List[str]
    confidence: float
    requires_followup: bool
```

### 8. Eligibility Checker

**Responsibilities:**
- Extract eligibility criteria from scheme documents
- Generate follow-up questions to gather user information
- Evaluate user qualification based on criteria
- Provide clear explanations for qualification status

**Interfaces:**

```python
class EligibilityChecker:
    def extract_criteria(self, scheme_name: str) -> List[EligibilityCriterion]
    def generate_questions(self, criteria: List[EligibilityCriterion]) -> List[str]
    def evaluate_eligibility(self, criteria: List[EligibilityCriterion], 
                            user_info: Dict[str, Any]) -> EligibilityResult

@dataclass
class EligibilityCriterion:
    criterion_id: str
    description: str
    criterion_type: str  # age, income, location, occupation, etc.
    required_value: Any
    comparison: str  # >, <, ==, in, etc.

@dataclass
class EligibilityResult:
    is_eligible: bool
    met_criteria: List[str]
    unmet_criteria: List[str]
    explanation: str
```

### 9. Text-to-Speech Service

**Responsibilities:**
- Convert text responses to audio in user's language
- Support 10 Indian languages + English
- Generate natural-sounding speech
- Provide audio controls (pause, replay, stop)

**Interfaces:**

```python
class TextToSpeechService:
    def synthesize(self, text: str, language_code: str, voice_gender: str = "female") -> AudioResult
    def get_available_voices(self, language_code: str) -> List[Voice]

@dataclass
class AudioResult:
    audio_data: bytes
    format: str  # mp3, wav, etc.
    duration: float
    error: Optional[str]

@dataclass
class Voice:
    voice_id: str
    language_code: str
    gender: str
    name: str
```

### 10. Scheme Database Manager

**Responsibilities:**
- Upload and process scheme PDF documents
- Extract text from PDFs
- Generate embeddings for document chunks
- Store embeddings in vector database with metadata
- Update and delete scheme documents

**Interfaces:**

```python
class SchemeDBManager:
    def upload_scheme(self, pdf_file: bytes, metadata: SchemeMetadata) -> str
    def process_pdf(self, pdf_file: bytes) -> List[str]  # Returns text chunks
    def generate_embeddings(self, chunks: List[str]) -> List[np.ndarray]
    def store_in_vector_db(self, embeddings: List[np.ndarray], 
                          chunks: List[str], metadata: SchemeMetadata) -> None
    def update_scheme(self, scheme_id: str, pdf_file: bytes) -> None
    def delete_scheme(self, scheme_id: str) -> None

@dataclass
class SchemeMetadata:
    scheme_id: str
    scheme_name: str
    category: str  # agriculture, healthcare, education, etc.
    ministry: str
    last_updated: datetime
    eligibility_summary: str
```

### 11. Analytics Service

**Responsibilities:**
- Log query metadata (language, category, response time)
- Track session metrics (duration, query count)
- Monitor error rates and types
- Track scheme access frequency
- Ensure no PII is logged

**Interfaces:**

```python
class AnalyticsService:
    def log_query(self, query_log: QueryLog) -> None
    def log_session(self, session_log: SessionLog) -> None
    def log_error(self, error_log: ErrorLog) -> None
    def track_scheme_access(self, scheme_id: str) -> None
    def get_analytics_report(self, start_date: datetime, end_date: datetime) -> AnalyticsReport

@dataclass
class QueryLog:
    session_id: str
    language_code: str
    scheme_category: str
    response_time_ms: int
    timestamp: datetime

@dataclass
class SessionLog:
    session_id: str
    duration_seconds: int
    query_count: int
    language_code: str
    timestamp: datetime

@dataclass
class ErrorLog:
    error_type: str
    error_message: str
    component: str
    timestamp: datetime
```

### 12. Offline Cache Manager

**Responsibilities:**
- Cache frequently accessed scheme information
- Store cached data in browser IndexedDB
- Queue queries when offline
- Sync queued queries when connection restored
- Manage cache size and expiration

**Interfaces:**

```typescript
interface OfflineCacheManager {
  cacheScheme(scheme: SchemeData): Promise<void>;
  getCachedScheme(schemeId: string): Promise<SchemeData | null>;
  queueQuery(query: Query): Promise<void>;
  syncQueuedQueries(): Promise<void>;
  isOnline(): boolean;
  clearExpiredCache(): Promise<void>;
}

interface SchemeData {
  schemeId: string;
  schemeName: string;
  summary: string;
  eligibility: string;
  benefits: string;
  cachedAt: Date;
}
```

## Data Models

### User Session Model

```python
class UserSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    language_code: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    conversation_history: List[Message] = []
    user_context: Dict[str, Any] = {}
    
    def is_expired(self) -> bool:
        return (datetime.utcnow() - self.last_activity).seconds > 1800  # 30 minutes
```

### Message Model

```python
class Message(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Literal["user", "assistant"]
    content: str
    language: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    audio_url: Optional[str] = None
```

### Scheme Document Model

```python
class SchemeDocument(BaseModel):
    scheme_id: str
    scheme_name: str
    category: str
    ministry: str
    description: str
    eligibility_criteria: List[str]
    benefits: List[str]
    application_process: str
    contact_info: str
    last_updated: datetime
    document_url: str
```

### Vector Embedding Model

```python
class VectorEmbedding(BaseModel):
    embedding_id: str
    scheme_id: str
    chunk_text: str
    embedding_vector: List[float]  # 768 or 1536 dimensions
    chunk_index: int
    metadata: Dict[str, Any]
```

### Query Request Model

```python
class QueryRequest(BaseModel):
    session_id: str
    audio_data: Optional[bytes] = None
    text_query: Optional[str] = None
    language_code: str
```

### Query Response Model

```python
class QueryResponse(BaseModel):
    response_text: str
    audio_url: str
    schemes: List[SchemeInfo]
    requires_followup: bool
    followup_questions: List[str] = []

class SchemeInfo(BaseModel):
    scheme_id: str
    scheme_name: str
    summary: str
    eligibility_summary: str
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Voice Input and Recording Properties

Property 1: Recording state machine correctness
*For any* voice input session, the system should transition correctly through states: pressing record activates recording, pressing stop immediately halts recording, 2+ seconds of silence automatically stops recording, and completion triggers transcription processing.
**Validates: Requirements 1.1, 1.3, 1.4, 1.5**

### Speech-to-Text Properties

Property 2: Transcription produces text output
*For any* valid audio input in a supported language, the transcription service should produce non-empty text output or a graceful error.
**Validates: Requirements 2.1, 2.3**

Property 3: Multi-language transcription support
*For any* of the 10 supported languages (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, English), the system should accept and successfully process transcription requests.
**Validates: Requirements 2.2**

Property 4: Transcription confirmation availability
*For any* successful transcription, the transcribed text should be available for user confirmation and display.
**Validates: Requirements 2.4**

### Translation Properties

Property 5: Query translation pipeline
*For any* transcribed text in a regional language, the system should translate it to English, forward to RAG_Engine, and handle translation failures with error logging and user notification.
**Validates: Requirements 3.1, 3.3, 3.4**

Property 6: Response translation to user language
*For any* English response generated by the LLM, the system should translate it to the user's preferred language, make it available for display, and preserve numbers and scheme details exactly.
**Validates: Requirements 7.1, 7.2, 7.3**

Property 7: Translation fallback to English
*For any* failed response translation, the system should display the original English text with an error notification.
**Validates: Requirements 7.4**

### RAG Engine Properties

Property 8: RAG pipeline execution
*For any* English query text, the RAG engine should generate an embedding, search the vector database, retrieve at most 5 document chunks, and forward them to the LLM.
**Validates: Requirements 4.1, 4.2, 4.3, 4.5**

### LLM Response Generation Properties

Property 9: Response generation from context
*For any* user query and retrieved document context, the LLM should generate a non-empty response that includes scheme names, key benefits, and basic eligibility criteria.
**Validates: Requirements 5.1, 5.5**

Property 10: Grounding in retrieved documents
*For any* LLM response, all factual claims should be traceable to information present in the retrieved scheme documents (no hallucination).
**Validates: Requirements 5.2**

Property 11: Insufficient information handling
*For any* query where retrieved documents lack sufficient information, the system should explicitly state that information is unavailable rather than generating speculative content.
**Validates: Requirements 5.3**

### Eligibility Checking Properties

Property 12: Eligibility assessment pipeline
*For any* scheme presented to the user, the eligibility checker should extract criteria, generate follow-up questions, evaluate user qualification, and provide clear status with explanations for unmet criteria.
**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**

### Text-to-Speech Properties

Property 13: TTS conversion and playback
*For any* translated response text, the voice interface should convert it to speech using the user's selected language, automatically initiate playback, and transition to ready state after completion.
**Validates: Requirements 8.1, 8.2, 8.3, 8.5**

### Conversation Context Properties

Property 14: Context preservation across turns
*For any* follow-up question in a session, the system should have access to all previous conversation messages, mentioned schemes, and user-provided information.
**Validates: Requirements 9.1, 9.2**

Property 15: Session timeout and context clearing
*For any* session with 30 or more minutes of inactivity or explicit user reset request, the system should clear conversation context and notify the user appropriately.
**Validates: Requirements 9.3, 9.4, 9.5**

### Language Selection Properties

Property 16: Multi-language support availability
*For any* language selection interface, all 10 supported languages (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, English) should be offered as options.
**Validates: Requirements 10.2**

Property 17: Language preference persistence and updates
*For any* language selection by the user, the preference should be stored for the session, and when changed, all UI text and voice interactions should update immediately.
**Validates: Requirements 10.3, 10.5**

### Offline Capability Properties

Property 18: Offline mode detection and data access
*For any* loss of internet connectivity, the system should detect offline state, display notification, allow browsing of cached scheme information, and clearly indicate unavailable features.
**Validates: Requirements 11.1, 11.2, 11.5**

Property 19: Query queuing and automatic sync
*For any* voice query submitted while offline, it should be queued and automatically processed when internet connection is restored.
**Validates: Requirements 11.3, 11.4**

### Data Privacy and Security Properties

Property 20: Data encryption and PII protection
*For any* data transmitted between client and server, HTTPS encryption should be used, and personally identifiable information should not be persisted beyond session lifetime or logged in any system logs.
**Validates: Requirements 12.1, 12.2, 12.5**

Property 21: Data cleanup on session end
*For any* session termination, all user-provided personal details and processed audio files should be deleted immediately.
**Validates: Requirements 12.3, 12.4**

### Performance Properties

Property 22: Pipeline component latency
*For any* query processing, transcription should complete within 3 seconds, document retrieval within 2 seconds, LLM response within 5 seconds, and TTS generation within 2 seconds.
**Validates: Requirements 13.1, 13.2, 13.3, 13.4**

Property 23: Initial load performance
*For any* application load on a 3G connection, the main interface should be displayed within 3 seconds.
**Validates: Requirements 13.5**

### Scheme Database Management Properties

Property 24: PDF processing and storage pipeline
*For any* valid PDF file containing scheme information, the system should accept upload, extract text, generate embeddings, and store them in the vector database with appropriate metadata.
**Validates: Requirements 14.1, 14.2, 14.3**

Property 25: Scheme update and deletion
*For any* scheme update operation, old embeddings should be replaced with new ones, and for any removal operation, all associated embeddings should be deleted from the vector database.
**Validates: Requirements 14.4, 14.5**

### Error Handling Properties

Property 26: Localized error messages
*For any* error that occurs during system operation, the error message displayed to the user should be in the user's selected language.
**Validates: Requirements 15.1**

Property 27: System overload and query understanding
*For any* system overload condition, the system should inform the user and suggest retrying later, and for any unprocessable query, provide example questions to guide the user.
**Validates: Requirements 15.3, 15.4**

Property 28: Dual-level error logging
*For any* technical error, the system should log detailed debugging information while displaying a simplified user-friendly message to the user.
**Validates: Requirements 15.5**

### Analytics Properties

Property 29: Query and session analytics
*For any* completed query, the system should log query language, scheme category, and response time without PII, and for any completed session, record duration and query count.
**Validates: Requirements 16.1, 16.2, 16.3**

Property 30: Error and scheme access tracking
*For any* error occurrence, the system should track error type and frequency, and for any scheme retrieval, track access patterns and frequency statistics.
**Validates: Requirements 16.4, 16.5**

### Mobile-First Interface Properties

Property 31: Responsive layout and touch optimization
*For any* mobile device with screen width between 320px and 768px, the application should display a properly adapted responsive layout with touch targets at least 44x44 pixels and font sizes at least 16px.
**Validates: Requirements 17.1, 17.2, 17.3**

Property 32: Sticky controls and collapsible content
*For any* scroll operation, primary action buttons should remain accessible, and scheme information should use collapsible sections on small screens.
**Validates: Requirements 17.4, 17.5**

### Accessibility Properties
### Accessibility Properties

Property 33: Screen reader and ARIA support
*For any* application load, all interface elements should be navigable using screen reader software with appropriate ARIA labels on interactive elements.
**Validates: Requirements 18.1, 18.2**

Property 34: Multi-modal feedback and text scaling
*For any* visual feedback provided, an equivalent audio cue should be provided, all text should support scaling up to 200% without breaking layout, and color-coded information should have text or icon alternatives.
**Validates: Requirements 18.3, 18.4, 18.5**

## Error Handling

### Error Categories and Handling Strategies

**1. Voice Input Errors**
- Microphone access denied → Display permission instructions in user's language
- Audio recording failure → Prompt user to check microphone and retry
- Background noise interference → Suggest quieter environment

**2. Transcription Errors**
- Speech-to-text API failure → Retry with exponential backoff, fallback to text input
- Unsupported language detected → Notify user and suggest supported languages
- Low confidence transcription → Show transcription for user confirmation/editing

**3. Translation Errors**
- Translation API unavailable → Queue request for retry, show English version
- Translation timeout → Retry once, then fallback to English with notification
- Unsupported language pair → Log error and notify development team

**4. RAG Pipeline Errors**
- Vector database connection failure → Return cached results if available, otherwise error message
- No relevant documents found → Inform user politely, suggest alternative queries
- Embedding generation failure → Retry once, then log error and notify user

**5. LLM Errors**
- API rate limit exceeded → Queue request and inform user of delay
- Response timeout → Retry with shorter context, inform user if still fails
- Inappropriate content detected → Filter response and log for review

**6. Network Errors**
- Connection loss during query → Save query to offline queue
- Slow connection → Show loading indicator, set appropriate timeouts
- Connection restored → Auto-sync queued queries

**7. Data Errors**
- Session expired → Clear context and start new session with notification
- Cache corruption → Clear cache and re-fetch data
- Invalid user input → Validate and provide specific error guidance

### Error Response Format

```python
@dataclass
class ErrorResponse:
    error_code: str
    error_message_en: str
    error_message_localized: str
    user_action: str  # What user should do next
    retry_possible: bool
    technical_details: Optional[str]  # For logging only
```

### Logging Strategy

**Error Logs:**
- Timestamp, error type, component, stack trace
- Request ID for tracing
- No PII (sanitize before logging)

**Performance Logs:**
- Latency metrics for each pipeline stage
- Resource utilization (CPU, memory, API calls)
- Cache hit/miss rates

**Analytics Logs:**
- Query patterns (language, category, time)
- User journey (session flow, drop-off points)
- Feature usage statistics

## Testing Strategy

### Dual Testing Approach

The testing strategy employs both unit testing and property-based testing as complementary approaches:

**Unit Tests** focus on:
- Specific examples demonstrating correct behavior
- Edge cases (empty inputs, boundary values, special characters)
- Error conditions and exception handling
- Integration points between components
- Mock external API responses

**Property-Based Tests** focus on:
- Universal properties that hold for all inputs
- Comprehensive input coverage through randomization
- State machine correctness
- Pipeline flow validation
- Data integrity across transformations

Both approaches are necessary for comprehensive coverage. Unit tests catch concrete bugs and validate specific scenarios, while property tests verify general correctness across a wide input space.

### Property-Based Testing Configuration

**Framework:** Hypothesis (Python) for backend services, fast-check (TypeScript) for frontend

**Configuration:**
- Minimum 100 iterations per property test (due to randomization)
- Each test tagged with: **Feature: nagrik-sahayak, Property {number}: {property_text}**
- Shrinking enabled to find minimal failing examples
- Seed-based reproducibility for CI/CD

**Example Property Test Structure:**

```python
from hypothesis import given, strategies as st
import pytest

@given(
    audio_data=st.binary(min_size=1000, max_size=100000),
    language=st.sampled_from(['hi', 'ta', 'te', 'bn', 'mr', 'gu', 'kn', 'ml', 'pa', 'en'])
)
@pytest.mark.property_test
def test_property_3_multilanguage_support(audio_data, language):
    """
    Feature: nagrik-sahayak, Property 3: Multi-language transcription support
    For any of the 10 supported languages, the system should accept 
    and successfully process transcription requests.
    """
    stt_service = SpeechToTextService()
    result = stt_service.transcribe(audio_data, language)
    
    # Should not raise exception
    assert result is not None
    # Should return either success or graceful error
    assert result.text or result.error
    # Should handle the requested language
    assert result.language_detected in SUPPORTED_LANGUAGES
```

### Unit Testing Strategy

**Test Coverage Goals:**
- 80%+ code coverage for business logic
- 100% coverage for critical paths (authentication, data privacy, payment flows if applicable)
- All error handlers must have explicit tests

**Key Unit Test Areas:**

1. **Voice Input Component**
   - Test recording start/stop
   - Test silence detection with various audio patterns
   - Test microphone permission handling

2. **Translation Service**
   - Test each language pair with sample sentences
   - Test number preservation (e.g., "₹50,000" remains unchanged)
   - Test special character handling

3. **RAG Engine**
   - Test embedding generation consistency
   - Test vector search ranking
   - Test empty result handling

4. **Eligibility Checker**
   - Test criteria extraction from sample schemes
   - Test evaluation logic with various user profiles
   - Test edge cases (missing criteria, ambiguous answers)

5. **Session Management**
   - Test session creation and expiration
   - Test context preservation across multiple turns
   - Test PII cleanup on session end

### Integration Testing

**End-to-End Flows:**
1. Complete voice query flow (voice → transcription → translation → RAG → LLM → translation → TTS)
2. Multi-turn conversation with context preservation
3. Eligibility checking flow with follow-up questions
4. Offline mode with query queuing and sync
5. Language switching mid-session

**API Integration Tests:**
- Mock external APIs (Bhashini, Gemini/GPT-4, Vector DB)
- Test retry logic and fallback mechanisms
- Test rate limiting and quota management

### Performance Testing

**Load Testing:**
- Simulate 100 concurrent users
- Test response times under load
- Identify bottlenecks in pipeline

**Latency Testing:**
- Measure each component's latency
- Validate against requirements (3s transcription, 2s retrieval, 5s LLM, 2s TTS)
- Test on simulated 3G connections

### Security Testing

**Privacy Tests:**
- Verify PII is not logged
- Verify session data is deleted
- Verify audio files are deleted after transcription
- Test HTTPS enforcement

**Penetration Testing:**
- Test for injection attacks (SQL, NoSQL, prompt injection)
- Test authentication and authorization
- Test rate limiting and DDoS protection

### Accessibility Testing

**Automated Tests:**
- Run axe-core or similar tool for WCAG compliance
- Test keyboard navigation
- Test screen reader compatibility (NVDA, JAWS)

**Manual Tests:**
- Test with actual screen readers
- Test with 200% text scaling
- Test with color blindness simulators
- Test touch target sizes on real devices

### Mobile Testing

**Device Testing:**
- Test on various screen sizes (320px to 768px)
- Test on different browsers (Chrome, Safari, Firefox)
- Test on different OS versions (Android 8+, iOS 13+)
- Test offline functionality
- Test PWA installation and updates

### Continuous Integration

**CI Pipeline:**
1. Lint and format checks
2. Unit tests (fast feedback)
3. Property-based tests (100 iterations)
4. Integration tests
5. Build and deploy to staging
6. Automated E2E tests on staging
7. Performance benchmarks

**Quality Gates:**
- All tests must pass
- Code coverage ≥ 80%
- No critical security vulnerabilities
- Performance benchmarks within thresholds


## Implementation Considerations

### Technology Selection Rationale

**Frontend Framework: React + TypeScript**
- Strong PWA support for offline capability
- Large ecosystem for mobile-first components
- TypeScript provides type safety for complex state management
- Excellent Web Speech API integration

**Backend Framework: FastAPI**
- High performance async support for concurrent requests
- Automatic API documentation (OpenAPI/Swagger)
- Native Pydantic integration for data validation
- Easy integration with Python ML libraries

**Vector Database: Pinecone vs ChromaDB**
- Pinecone: Managed service, better for production scale, higher cost
- ChromaDB: Self-hosted, lower cost, good for MVP and development
- Recommendation: Start with ChromaDB for MVP, migrate to Pinecone for production scale

**LLM Provider: Google Gemini vs OpenAI GPT-4**
- Gemini: Better multilingual support, competitive pricing, Google Cloud integration
- GPT-4: More mature ecosystem, better documentation, higher cost
- Recommendation: Gemini for production (better Indian language support)

**Speech Services: Bhashini API**
- Government-backed initiative for Indian languages
- Free tier available for public good applications
- Supports all 10 target languages
- Fallback to OpenAI Whisper for STT if needed

### Deployment Architecture

**Development Environment:**
- Local Docker Compose setup
- Mock external APIs for offline development
- SQLite for local vector storage

**Staging Environment:**
- AWS ECS or GCP Cloud Run for containerized services
- Managed Redis (ElastiCache/Cloud Memorystore)
- Managed PostgreSQL (RDS/Cloud SQL)
- ChromaDB or Pinecone for vector storage

**Production Environment:**
- Multi-region deployment for low latency
- CDN for static assets (CloudFront/Cloud CDN)
- Auto-scaling based on load
- Monitoring with CloudWatch/Cloud Monitoring
- Logging with ELK stack or Cloud Logging

### Security Considerations

**Authentication & Authorization:**
- No user authentication required for MVP (public access)
- Rate limiting per IP address to prevent abuse
- API key authentication for admin endpoints

**Data Protection:**
- HTTPS/TLS 1.3 for all communications
- Encryption at rest for cached data
- Regular security audits and penetration testing
- GDPR/data privacy compliance

**API Security:**
- Input validation and sanitization
- Protection against prompt injection attacks
- Rate limiting on external API calls
- Secrets management (AWS Secrets Manager/GCP Secret Manager)

### Scalability Considerations

**Horizontal Scaling:**
- Stateless API servers (scale with load)
- Redis for distributed session management
- Vector DB handles concurrent queries
- LLM API calls can be parallelized

**Performance Optimization:**
- Response caching for common queries
- Embedding caching to reduce computation
- Lazy loading for mobile UI
- Progressive Web App for offline capability
- CDN for static assets

**Cost Optimization:**
- Cache frequently accessed schemes
- Batch embedding generation for new schemes
- Use smaller LLM models for simple queries
- Monitor and optimize API usage

### Monitoring and Observability

**Metrics to Track:**
- API response times (p50, p95, p99)
- Error rates by component
- External API latency and failures
- Cache hit/miss rates
- User engagement (queries per session, session duration)

**Alerting:**
- High error rates (>5%)
- Slow response times (>10s)
- External API failures
- High resource utilization (>80% CPU/memory)

**Logging:**
- Structured JSON logs
- Request/response tracing with correlation IDs
- Error stack traces (sanitized of PII)
- Performance metrics per pipeline stage

### Development Workflow

**Version Control:**
- Git with feature branch workflow
- Pull request reviews required
- Automated CI/CD pipeline

**Testing Strategy:**
- Unit tests run on every commit
- Property-based tests in CI pipeline
- Integration tests on staging deployment
- Manual accessibility testing before release

**Release Process:**
- Semantic versioning (MAJOR.MINOR.PATCH)
- Blue-green deployment for zero downtime
- Gradual rollout with feature flags
- Rollback plan for failed deployments

### Future Enhancements

**Phase 2 Features:**
- User accounts and personalized recommendations
- Application form pre-filling assistance
- Document upload for eligibility verification
- SMS/WhatsApp integration for wider reach

**Phase 3 Features:**
- Integration with government portals for direct application
- Real-time scheme updates via webhooks
- Community features (success stories, Q&A)
- Advanced analytics and insights dashboard

**Technical Improvements:**
- Fine-tuned LLM for Indian government schemes
- Custom speech models for regional accents
- Edge deployment for ultra-low latency
- Blockchain for audit trail of scheme information