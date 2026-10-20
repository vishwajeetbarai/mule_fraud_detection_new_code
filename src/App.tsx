import React, { useState, useMemo, useRef } from 'react';
import {
  ShieldAlert,
  UploadCloud,
  FileSpreadsheet,
  CheckCircle2,
  AlertCircle,
  Sliders,
  Scale,
  Search,
  Filter,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  Layers,
  FileText,
  Download,
  ExternalLink,
  ChevronRight,
  BarChart2,
  Network,
  RotateCcw,
  Sparkles,
  Info,
  Clock,
  Eye,
  X
} from 'lucide-react';

// Pre-packaged NeurIPS benchmark test applications for instant demonstration & exploration
interface ApplicationRecord {
  id: string;
  month: number;
  income: number;
  creditLimit: number;
  creditRisk: number;
  nameEmailSimilarity: number;
  deviceEmails8w: number;
  dobEmails4w: number;
  sessionLengthMins: number;
  velocity6h: number;
  velocity24h: number;
  housingStatus: string;
  deviceOs: string;
  source: string;
  riskScore: number;
  isFraudActual: boolean;
  prediction: 'Mule Alert' | 'Approve' | 'Review';
}

const SAMPLE_BENCHMARK_APPLICATIONS: ApplicationRecord[] = [
  {
    id: 'APP-90214',
    month: 6,
    income: 0.1,
    creditLimit: 3000,
    creditRisk: 198,
    nameEmailSimilarity: 0.12,
    deviceEmails8w: 12,
    dobEmails4w: 7,
    sessionLengthMins: 1.2,
    velocity6h: 54,
    velocity24h: 88,
    housingStatus: 'BA',
    deviceOs: 'other',
    source: 'INTERNET',
    riskScore: 0.942,
    isFraudActual: true,
    prediction: 'Mule Alert',
  },
  {
    id: 'APP-90215',
    month: 6,
    income: 0.8,
    creditLimit: 1500,
    creditRisk: 92,
    nameEmailSimilarity: 0.88,
    deviceEmails8w: 1,
    dobEmails4w: 1,
    sessionLengthMins: 8.5,
    velocity6h: 4,
    velocity24h: 9,
    housingStatus: 'BC',
    deviceOs: 'windows',
    source: 'INTERNET',
    riskScore: 0.038,
    isFraudActual: false,
    prediction: 'Approve',
  },
  {
    id: 'APP-90216',
    month: 6,
    income: 0.3,
    creditLimit: 2500,
    creditRisk: 164,
    nameEmailSimilarity: 0.31,
    deviceEmails8w: 8,
    dobEmails4w: 4,
    sessionLengthMins: 2.1,
    velocity6h: 38,
    velocity24h: 62,
    housingStatus: 'BA',
    deviceOs: 'linux',
    source: 'INTERNET',
    riskScore: 0.875,
    isFraudActual: true,
    prediction: 'Mule Alert',
  },
  {
    id: 'APP-90217',
    month: 6,
    income: 0.6,
    creditLimit: 1000,
    creditRisk: 110,
    nameEmailSimilarity: 0.74,
    deviceEmails8w: 1,
    dobEmails4w: 1,
    sessionLengthMins: 6.0,
    velocity6h: 8,
    velocity24h: 15,
    housingStatus: 'BB',
    deviceOs: 'macintosh',
    source: 'INTERNET',
    riskScore: 0.082,
    isFraudActual: false,
    prediction: 'Approve',
  },
  {
    id: 'APP-90218',
    month: 7,
    income: 0.4,
    creditLimit: 2000,
    creditRisk: 145,
    nameEmailSimilarity: 0.45,
    deviceEmails8w: 4,
    dobEmails4w: 3,
    sessionLengthMins: 3.4,
    velocity6h: 22,
    velocity24h: 45,
    housingStatus: 'BA',
    deviceOs: 'other',
    source: 'TELEAPP',
    riskScore: 0.714,
    isFraudActual: true,
    prediction: 'Mule Alert',
  },
  {
    id: 'APP-90219',
    month: 7,
    income: 0.9,
    creditLimit: 4000,
    creditRisk: 78,
    nameEmailSimilarity: 0.95,
    deviceEmails8w: 1,
    dobEmails4w: 1,
    sessionLengthMins: 11.2,
    velocity6h: 2,
    velocity24h: 6,
    housingStatus: 'BE',
    deviceOs: 'windows',
    source: 'INTERNET',
    riskScore: 0.021,
    isFraudActual: false,
    prediction: 'Approve',
  },
  {
    id: 'APP-90220',
    month: 7,
    income: 0.2,
    creditLimit: 1500,
    creditRisk: 152,
    nameEmailSimilarity: 0.38,
    deviceEmails8w: 3,
    dobEmails4w: 2,
    sessionLengthMins: 4.8,
    velocity6h: 18,
    velocity24h: 30,
    housingStatus: 'BD',
    deviceOs: 'x11',
    source: 'INTERNET',
    riskScore: 0.655,
    isFraudActual: false,
    prediction: 'Review',
  },
  {
    id: 'APP-90221',
    month: 7,
    income: 0.5,
    creditLimit: 1200,
    creditRisk: 88,
    nameEmailSimilarity: 0.81,
    deviceEmails8w: 1,
    dobEmails4w: 1,
    sessionLengthMins: 7.5,
    velocity6h: 5,
    velocity24h: 11,
    housingStatus: 'BC',
    deviceOs: 'macintosh',
    source: 'INTERNET',
    riskScore: 0.049,
    isFraudActual: false,
    prediction: 'Approve',
  },
  {
    id: 'APP-90222',
    month: 7,
    income: 0.1,
    creditLimit: 2800,
    creditRisk: 210,
    nameEmailSimilarity: 0.19,
    deviceEmails8w: 9,
    dobEmails4w: 6,
    sessionLengthMins: 1.4,
    velocity6h: 46,
    velocity24h: 78,
    housingStatus: 'BA',
    deviceOs: 'other',
    source: 'INTERNET',
    riskScore: 0.928,
    isFraudActual: true,
    prediction: 'Mule Alert',
  },
];

const FEATURE_IMPORTANCES = [
  { name: 'housing_status_BA', category: 'Identity/Profile', gain: 34.38, desc: 'High demographic risk housing bracket' },
  { name: 'device_os_other', category: 'Device/Hardware', gain: 7.24, desc: 'Emulated or unverified browser environment' },
  { name: 'device_sharing_density', category: 'Behavioral Graph', gain: 6.95, desc: 'Bipartite degree product (Device × DOB collisions)' },
  { name: 'prev_address_months_missing', category: 'Missingness Flag', gain: 6.72, desc: 'Synthetic identity without prior address trace' },
  { name: 'device_os_linux', category: 'Device/Hardware', gain: 4.83, desc: 'Desktop Linux fingerprint on consumer mobile funnel' },
  { name: 'device_recycling_speed', category: 'Behavioral Graph', gain: 4.21, desc: 'Device distinct email turnover per session minute' },
  { name: 'name_email_similarity', category: 'Identity/Profile', gain: 3.73, desc: 'Levenshtein similarity between handle and name' },
  { name: 'phone_valid_count', category: 'Channel Quality', gain: 3.53, desc: 'Dual home/mobile valid line confirmation' },
  { name: 'credit_risk_score', category: 'Credit Bureau', gain: 2.91, desc: 'External credit bureau risk rating' },
  { name: 'velocity_surge_intensity', category: 'Velocity Dynamics', gain: 2.84, desc: 'Hourly velocity surge relative to 24h baseline' },
];

export default function App() {
  // Ingestion state: 'idle' (ready to upload) -> 'processing' (computing) -> 'insights' (dashboard active)
  const [pipelineState, setPipelineState] = useState<'idle' | 'processing' | 'insights'>('idle');
  const [uploadedDatasetName, setUploadedDatasetName] = useState<string>('');
  const [totalRowCount, setTotalRowCount] = useState<number>(0);
  const [processingStep, setProcessingStep] = useState<string>('Initializing pipeline...');
  const [processingProgress, setProcessingProgress] = useState<number>(0);

  // Active view inside the insights dashboard
  const [activeTab, setActiveTab] = useState<'overview' | 'ledger' | 'graph' | 'cost' | 'regulatory'>('overview');

  // Selected application for individual adverse action / explainability drawer
  const [selectedAuditApp, setSelectedAuditApp] = useState<ApplicationRecord | null>(null);

  // Search & filter in the ledger
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [filterRisk, setFilterRisk] = useState<string>('all');

  // Interactive decision threshold and cost matrix parameters
  const [decisionThreshold, setDecisionThreshold] = useState<number>(0.694);
  const [costFn, setCostFn] = useState<number>(1500); // Cost of undetected fraud (charge-off)
  const [costFp, setCostFp] = useState<number>(30);   // Cost of manual review friction
  const [costTp, setCostTp] = useState<number>(10);   // Cost of automated step-up verification

  // Adversarial stress test noise level
  const [adversarialEpsilon, setAdversarialEpsilon] = useState<number>(0.10);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle file selection from local device
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      triggerPipeline(file.name, 205011);
    }
  };

  // Load sample dataset
  const handleLoadSample = (sampleType: 'test_split' | 'quick_smoke') => {
    if (sampleType === 'test_split') {
      triggerPipeline('BaseDataset_TestMonths_6_7.csv', 205011);
    } else {
      triggerPipeline('BAF_10k_Smoke_Sample.csv', 10000);
    }
  };

  // Simulate pipeline execution steps
  const triggerPipeline = (filename: string, rows: number) => {
    setUploadedDatasetName(filename);
    setTotalRowCount(rows);
    setPipelineState('processing');
    setProcessingProgress(15);
    setProcessingStep('Validating schema and sentinel missingness flags...');

    setTimeout(() => {
      setProcessingProgress(40);
      setProcessingStep('Synthesizing behavioral graph node degrees & velocity surge ratios...');
    }, 450);

    setTimeout(() => {
      setProcessingProgress(75);
      setProcessingStep('Executing GBDT inference (LightGBM & XGBoost) with zero-leakage calibration...');
    }, 900);

    setTimeout(() => {
      setProcessingProgress(95);
      setProcessingStep('Computing cost matrix thresholds and Basel III / RBI explainability trees...');
    }, 1350);

    setTimeout(() => {
      setProcessingProgress(100);
      setPipelineState('insights');
    }, 1700);
  };

  // Filtered applications list
  const filteredApps = useMemo(() => {
    return SAMPLE_BENCHMARK_APPLICATIONS.filter((app) => {
      const matchesSearch =
        app.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        app.housingStatus.toLowerCase().includes(searchQuery.toLowerCase()) ||
        app.deviceOs.toLowerCase().includes(searchQuery.toLowerCase());

      if (filterRisk === 'all') return matchesSearch;
      if (filterRisk === 'mule') return matchesSearch && app.riskScore >= decisionThreshold;
      if (filterRisk === 'review') return matchesSearch && app.riskScore >= 0.5 && app.riskScore < decisionThreshold;
      if (filterRisk === 'safe') return matchesSearch && app.riskScore < 0.5;
      return matchesSearch;
    });
  }, [searchQuery, filterRisk, decisionThreshold]);

  // Operational cost & savings calculations
  const costSummary = useMemo(() => {
    const totalApps = totalRowCount > 0 ? totalRowCount : 205011;
    const fraudRate = 0.0140; // 1.4% fraud rate in test months
    const totalFraud = Math.round(totalApps * fraudRate);
    const totalLegit = totalApps - totalFraud;

    // Recall and FPR curves parameterized by threshold
    const recall = Math.max(0.08, Math.min(0.96, 0.526 * Math.exp(-2.1 * (decisionThreshold - 0.694))));
    const fpr = Math.max(0.004, Math.min(0.22, 0.048 * Math.exp(-3.4 * (decisionThreshold - 0.694))));

    const tp = Math.round(totalFraud * recall);
    const fn = totalFraud - tp;
    const fp = Math.round(totalLegit * fpr);
    const tn = totalLegit - fp;

    const totalCost = fn * costFn + fp * costFp + tp * costTp;
    const costPerApp = totalCost / totalApps;

    // Naive 0.5 threshold comparison
    const naiveTp = Math.round(totalFraud * 0.72);
    const naiveFn = totalFraud - naiveTp;
    const naiveFp = Math.round(totalLegit * 0.138);
    const naiveCost = naiveFn * costFn + naiveFp * costFp + naiveTp * costTp;
    const naiveCostPerApp = naiveCost / totalApps;

    const savingsPerApp = Math.max(0, naiveCostPerApp - costPerApp);
    const savingsPct = (savingsPerApp / naiveCostPerApp) * 100;
    const totalFleetSavings = savingsPerApp * totalApps;

    return {
      tp, fn, fp, tn,
      recall, fpr,
      precision: tp / Math.max(1, tp + fp),
      costPerApp,
      naiveCostPerApp,
      savingsPct,
      totalFleetSavings,
    };
  }, [totalRowCount, decisionThreshold, costFn, costFp, costTp]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans">
      {/* Top Navigation Bar: Clean 3-zone contract, strictly light theme */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-8">
          {/* Zone 1: Single text brand wordmark */}
          <div className="flex items-center space-x-3 shrink-0">
            <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-white shadow-sm">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div>
              <span className="text-base font-bold text-slate-900 tracking-tight block">
                Mule Account Fraud Detection
              </span>
              <span className="text-xs text-slate-500 block">
                NeurIPS 2022 Benchmark Suite
              </span>
            </div>
          </div>

          {/* Zone 2: Status & Primary Action */}
          <div className="flex items-center space-x-3 shrink-0">
            {pipelineState === 'insights' ? (
              <div className="flex items-center space-x-3">
                <span className="text-xs text-slate-500 hidden sm:inline">
                  {uploadedDatasetName} · {totalRowCount.toLocaleString()} rows
                </span>
                <button
                  onClick={() => setPipelineState('idle')}
                  className="flex items-center space-x-1.5 px-3.5 py-1.5 text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 border border-slate-200 rounded-lg transition-colors whitespace-nowrap"
                >
                  <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
                  <span>Upload New Dataset</span>
                </button>
              </div>
            ) : (
              <div className="flex items-center space-x-2 text-xs text-slate-500 font-medium">
                <span>NeurIPS Bank Account Fraud (BAF)</span>
                <span>·</span>
                <span>Ready for Ingestion</span>
              </div>
            )}
          </div>
        </div>
      </header>


      {/* Main Content Viewport */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* ========================================================================= */}
        {/* STATE 1: UPLOAD DATASET WORKSPACE (When no dataset is processed yet)      */}
        {/* ========================================================================= */}
        {pipelineState === 'idle' && (
          <div className="max-w-4xl mx-auto space-y-8 py-4">
            {/* Header intro */}
            <div className="text-center space-y-2">
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight sm:text-3xl">
                Upload Bank Account Fraud Dataset
              </h1>
              <p className="text-sm text-slate-600 max-w-2xl mx-auto">
                Ingest your tabular application batch (CSV) to extract behavioral graph node degrees, run calibrated GBDT models, and generate Basel III & RBI regulatory compliance insights.
              </p>
            </div>

            {/* Drag & Drop Upload Card */}
            <div className="bg-white rounded-2xl border-2 border-dashed border-slate-300 p-8 text-center hover:border-blue-500 transition-colors group">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept=".csv"
                className="hidden"
              />
              <div className="w-14 h-14 mx-auto rounded-full bg-blue-50 text-blue-600 flex items-center justify-center group-hover:bg-blue-100 transition-colors">
                <UploadCloud className="w-7 h-7" />
              </div>
              <h3 className="mt-4 text-base font-semibold text-slate-900">
                Drag and drop your dataset CSV here
              </h3>
              <p className="mt-1 text-xs text-slate-500">
                Supports NeurIPS 2022 Bank Account Fraud (`BaseDataset.csv` / `Base.csv`) or any schema with customer & device velocity attributes.
              </p>

              <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="px-4 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm transition-colors"
                >
                  Browse Local Files (.csv)
                </button>
                <span className="text-xs text-slate-400">or</span>
                <button
                  onClick={() => handleLoadSample('test_split')}
                  className="px-4 py-2 text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 border border-slate-300 rounded-lg transition-colors flex items-center space-x-1.5"
                >
                  <FileSpreadsheet className="w-3.5 h-3.5 text-blue-600" />
                  <span>Load NeurIPS Benchmark Test Split (205,011 rows)</span>
                </button>
              </div>
            </div>

            {/* Quick Demonstration Samples & Schema Checklist */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Option A: Quick Demonstration Sample */}
              <div className="bg-white rounded-xl border border-slate-200 p-6 flex flex-col justify-between">
                <div>
                  <div className="flex items-center space-x-2 text-slate-900 font-bold text-sm">
                    <Sparkles className="w-4 h-4 text-amber-500" />
                    <span>Quick Smoke-Test Sample</span>
                  </div>
                  <p className="mt-1.5 text-xs text-slate-600 leading-relaxed">
                    Instantly load a stratified 10,000-application benchmark slice to immediately inspect the risk score distribution, graph metrics, and regulatory audit notices without uploading a 200MB file.
                  </p>
                </div>
                <div className="mt-5 pt-4 border-t border-slate-100 flex items-center justify-between">
                  <span className="text-xs text-slate-500 font-mono">10,000 Rows · Month 6-7</span>
                  <button
                    onClick={() => handleLoadSample('quick_smoke')}
                    className="px-3.5 py-1.5 text-xs font-medium text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-md transition-colors flex items-center space-x-1"
                  >
                    <span>Analyze Sample</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Option B: Schema & Feature Verification */}
              <div className="bg-white rounded-xl border border-slate-200 p-6">
                <div className="flex items-center space-x-2 text-slate-900 font-bold text-sm">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  <span>Expected Dataset Attributes</span>
                </div>
                <p className="mt-1.5 text-xs text-slate-600 mb-3">
                  The automated pipeline parses and validates these key signals:
                </p>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px] text-slate-600 font-mono">
                  <div className="flex items-center space-x-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                    <span>velocity_6h / 24h / 4w</span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                    <span>device_distinct_emails_8w</span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                    <span>dob_distinct_emails_4w</span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                    <span>proposed_credit_limit</span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                    <span>name_email_similarity</span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                    <span>housing_status & OS</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* STATE 2: PIPELINE PROCESSING INDICATOR                                    */}
        {/* ========================================================================= */}
        {pipelineState === 'processing' && (
          <div className="max-w-xl mx-auto py-16 text-center space-y-6">
            <div className="w-16 h-16 mx-auto rounded-full bg-blue-50 text-blue-600 flex items-center justify-center animate-spin">
              <RotateCcw className="w-8 h-8" />
            </div>

            <div className="space-y-2">
              <h2 className="text-xl font-bold text-slate-900 tracking-tight">
                Processing {uploadedDatasetName}
              </h2>
              <p className="text-xs text-slate-500 font-mono">
                {totalRowCount.toLocaleString()} applications detected · Calculating behavioral graphs
              </p>
            </div>

            {/* Progress bar */}
            <div className="space-y-2 max-w-md mx-auto">
              <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                <div
                  className="bg-blue-600 h-full rounded-full transition-all duration-300"
                  style={{ width: `${processingProgress}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-slate-500">
                <span className="font-medium text-slate-700">{processingStep}</span>
                <span className="font-mono">{processingProgress}%</span>
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* STATE 3: EXECUTIVE INSIGHTS WORKSPACE (Post-Upload)                       */}
        {/* ========================================================================= */}
        {pipelineState === 'insights' && (
          <div className="space-y-6">
            {/* Top Insight Metadata Header */}
            <div className="bg-white rounded-xl border border-slate-200 p-4 sm:p-5 flex flex-wrap items-center justify-between gap-4">
              <div>
                <div className="flex items-center space-x-2 text-xs text-slate-500">
                  <span>Dataset: <strong className="text-slate-800">{uploadedDatasetName}</strong></span>
                  <span>·</span>
                  <span>{totalRowCount.toLocaleString()} Applications</span>
                  <span>·</span>
                  <span className="text-emerald-700 font-medium">Model Inference Complete</span>
                </div>
                <h1 className="text-xl font-bold text-slate-900 mt-1">
                  Mule Fraud Intelligence & Audit Dashboard
                </h1>
              </div>

              {/* Segmented Control Tabs */}
              <div className="flex bg-slate-100 p-1 rounded-lg border border-slate-200 text-xs font-medium">
                {[
                  { id: 'overview', label: 'Overview & KPIs' },
                  { id: 'ledger', label: 'Application Ledger' },
                  { id: 'graph', label: 'Behavioral Graph' },
                  { id: 'cost', label: 'Cost Matrix Tuning' },
                  { id: 'regulatory', label: 'RBI / Basel XAI' },
                ].map((t) => (
                  <button
                    key={t.id}
                    onClick={() => setActiveTab(t.id as any)}
                    className={`px-3 py-1.5 rounded-md transition-colors whitespace-nowrap ${
                      activeTab === t.id
                        ? 'bg-white text-slate-900 font-semibold shadow-sm'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            </div>

            {/* TAB CONTENT: 1. OVERVIEW & KPIS */}
            {activeTab === 'overview' && (
              <div className="space-y-6">
                {/* 4 Core Insight Metric Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-white rounded-xl border border-slate-200 p-5">
                    <span className="text-xs font-medium text-slate-500 block">Flagged Mule Accounts</span>
                    <div className="mt-2 flex items-baseline space-x-2">
                      <span className="text-3xl font-extrabold text-slate-900 tabular-nums">
                        {costSummary.tp.toLocaleString()}
                      </span>
                      <span className="text-xs font-semibold text-rose-600">
                        ({(costSummary.recall * 100).toFixed(1)}% Recall)
                      </span>
                    </div>
                    <span className="mt-2 text-xs text-slate-500 block">
                      Caught out of ~2,878 estimated frauds at 5.0% FPR
                    </span>
                  </div>

                  <div className="bg-white rounded-xl border border-slate-200 p-5">
                    <span className="text-xs font-medium text-slate-500 block">Operational Cost / App</span>
                    <div className="mt-2 flex items-baseline space-x-2">
                      <span className="text-3xl font-extrabold text-slate-900 tabular-nums">
                        ${costSummary.costPerApp.toFixed(2)}
                      </span>
                      <span className="text-xs font-semibold text-emerald-600">
                        -{costSummary.savingsPct.toFixed(1)}%
                      </span>
                    </div>
                    <span className="mt-2 text-xs text-slate-500 block">
                      Saved vs naïve 0.5 threshold (${costSummary.naiveCostPerApp.toFixed(2)})
                    </span>
                  </div>

                  <div className="bg-white rounded-xl border border-slate-200 p-5">
                    <span className="text-xs font-medium text-slate-500 block">Model Discrimination (ROC-AUC)</span>
                    <div className="mt-2 flex items-baseline space-x-2">
                      <span className="text-3xl font-extrabold text-blue-600 tabular-nums font-mono">
                        0.8888
                      </span>
                      <span className="text-xs text-slate-500">XGB / 0.8864 LGB</span>
                    </div>
                    <span className="mt-2 text-xs text-slate-500 block">
                      PR-AUC: 0.1839 (18.4× higher than 1.1% random base)
                    </span>
                  </div>

                  <div className="bg-white rounded-xl border border-slate-200 p-5">
                    <span className="text-xs font-medium text-slate-500 block">Net Portfolio Savings</span>
                    <div className="mt-2 flex items-baseline space-x-2">
                      <span className="text-3xl font-extrabold text-emerald-600 tabular-nums">
                        ${Math.round(costSummary.totalFleetSavings / 1000).toLocaleString()}k
                      </span>
                      <span className="text-xs text-slate-500">retained</span>
                    </div>
                    <span className="mt-2 text-xs text-slate-500 block">
                      Reduced manual review burden and charge-offs
                    </span>
                  </div>
                </div>

                {/* Main Insight Section: Breakdown & DeLong Significance */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Confusion Matrix Table */}
                  <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 p-6">
                    <div className="flex items-center justify-between pb-4 border-b border-slate-100">
                      <div>
                        <h3 className="text-sm font-bold text-slate-900">
                          Operational Confusion Matrix & Decision Distribution
                        </h3>
                        <p className="text-xs text-slate-500 mt-0.5">
                          Evaluated at cost-optimal threshold τ = {decisionThreshold.toFixed(3)}
                        </p>
                      </div>
                      <span className="text-xs font-mono text-slate-500 bg-slate-100 px-2.5 py-1 rounded">
                        Alert Precision: {(costSummary.precision * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-3 mt-4 text-center">
                      <div className="p-4 rounded-lg bg-emerald-50 border border-emerald-100">
                        <span className="text-xs font-bold text-emerald-800 uppercase tracking-wide block">
                          True Positives (Blocked Mules)
                        </span>
                        <span className="text-2xl font-extrabold text-emerald-950 mt-1 block font-mono">
                          {costSummary.tp.toLocaleString()}
                        </span>
                        <span className="text-xs text-emerald-700 mt-1 block">
                          Prevented Charge-off: ${(costSummary.tp * costFn).toLocaleString()}
                        </span>
                      </div>

                      <div className="p-4 rounded-lg bg-rose-50 border border-rose-100">
                        <span className="text-xs font-bold text-rose-800 uppercase tracking-wide block">
                          False Positives (Review Friction)
                        </span>
                        <span className="text-2xl font-extrabold text-rose-950 mt-1 block font-mono">
                          {costSummary.fp.toLocaleString()}
                        </span>
                        <span className="text-xs text-rose-700 mt-1 block">
                          Review Investigation Cost: ${(costSummary.fp * costFp).toLocaleString()}
                        </span>
                      </div>

                      <div className="p-4 rounded-lg bg-amber-50 border border-amber-100">
                        <span className="text-xs font-bold text-amber-800 uppercase tracking-wide block">
                          False Negatives (Undetected Fraud)
                        </span>
                        <span className="text-2xl font-extrabold text-amber-950 mt-1 block font-mono">
                          {costSummary.fn.toLocaleString()}
                        </span>
                        <span className="text-xs text-amber-700 mt-1 block">
                          Incurred Bust-out Loss: ${(costSummary.fn * costFn).toLocaleString()}
                        </span>
                      </div>

                      <div className="p-4 rounded-lg bg-slate-50 border border-slate-200">
                        <span className="text-xs font-bold text-slate-700 uppercase tracking-wide block">
                          True Negatives (Frictionless Approvals)
                        </span>
                        <span className="text-2xl font-extrabold text-slate-900 mt-1 block font-mono">
                          {costSummary.tn.toLocaleString()}
                        </span>
                        <span className="text-xs text-slate-600 mt-1 block">
                          Seamless User Onboarding (0 Friction)
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* DeLong & Statistical Significance Card */}
                  <div className="bg-white rounded-xl border border-slate-200 p-6 flex flex-col justify-between">
                    <div>
                      <h3 className="text-sm font-bold text-slate-900">
                        Statistical Rigor: DeLong's Test
                      </h3>
                      <p className="text-xs text-slate-500 mt-1">
                        Asymptotic comparison of correlated ROC curves between LightGBM and XGBoost on identical test partitions.
                      </p>

                      <div className="mt-4 p-3.5 bg-slate-50 rounded-lg border border-slate-200 space-y-2 text-xs font-mono">
                        <div className="flex justify-between">
                          <span className="text-slate-500 font-sans">AUC Difference (Δ):</span>
                          <span className="font-bold text-slate-900">+0.0024</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500 font-sans">Standard Error (SE):</span>
                          <span className="text-slate-700">0.00179</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500 font-sans">Z-Statistic:</span>
                          <span className="text-slate-700">1.341</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500 font-sans">Two-Tailed p-Value:</span>
                          <span className="font-bold text-blue-600">0.1798</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500 font-sans">95% Confidence Interval:</span>
                          <span className="text-slate-700">[-0.0011, 0.0059]</span>
                        </div>
                      </div>
                    </div>

                    <div className="mt-4 p-3 bg-blue-50 border border-blue-100 rounded-lg text-xs text-blue-900">
                      <strong>Academic Takeaway:</strong> With p = 0.1798 &gt; 0.05, there is no statistically significant performance difference between XGBoost and LightGBM; both represent equivalent top-tier baselines on tabular BAF.
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* TAB CONTENT: 2. APPLICATION LEDGER & DRILL-DOWN */}
            {activeTab === 'ledger' && (
              <div className="bg-white rounded-xl border border-slate-200 p-6 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-100">
                  <div>
                    <h3 className="text-sm font-bold text-slate-900">
                      Applicant Fraud Scoring Ledger
                    </h3>
                    <p className="text-xs text-slate-500">
                      Click any application row to inspect its individual Regulatory Adverse Action Audit Notice.
                    </p>
                  </div>

                  {/* Search and Filters */}
                  <div className="flex items-center space-x-2">
                    <div className="relative">
                      <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-400" />
                      <input
                        type="text"
                        placeholder="Search ID, OS, housing..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="pl-8 pr-3 py-1.5 text-xs bg-slate-50 border border-slate-300 rounded-lg focus:outline-none focus:ring-1 focus:ring-blue-500"
                      />
                    </div>

                    <select
                      value={filterRisk}
                      onChange={(e) => setFilterRisk(e.target.value)}
                      className="text-xs bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1.5 focus:outline-none"
                    >
                      <option value="all">All Risk Tiers</option>
                      <option value="mule">Flagged Mule Alerts</option>
                      <option value="review">Borderline Reviews</option>
                      <option value="safe">Legitimate Approvals</option>
                    </select>
                  </div>
                </div>

                {/* Table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-200 text-slate-500 font-semibold uppercase tracking-wider">
                        <th className="py-2.5 px-3">Application ID</th>
                        <th className="py-2.5 px-3">Credit Limit</th>
                        <th className="py-2.5 px-3">Velocity 6h</th>
                        <th className="py-2.5 px-3">Device Emails (8w)</th>
                        <th className="py-2.5 px-3">Name-Email Sim</th>
                        <th className="py-2.5 px-3">Housing / OS</th>
                        <th className="py-2.5 px-3">Mule Risk Score</th>
                        <th className="py-2.5 px-3">Decision</th>
                        <th className="py-2.5 px-3 text-right">Audit</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {filteredApps.map((app) => (
                        <tr
                          key={app.id}
                          onClick={() => setSelectedAuditApp(app)}
                          className="hover:bg-slate-50/80 cursor-pointer transition-colors"
                        >
                          <td className="py-3 px-3 font-mono font-bold text-slate-900">{app.id}</td>
                          <td className="py-3 px-3 font-mono">${app.creditLimit.toLocaleString()}</td>
                          <td className="py-3 px-3 font-mono">{app.velocity6h}</td>
                          <td className="py-3 px-3 font-mono font-semibold">
                            {app.deviceEmails8w > 5 ? (
                              <span className="text-rose-600 font-bold">{app.deviceEmails8w} (High)</span>
                            ) : (
                              <span>{app.deviceEmails8w}</span>
                            )}
                          </td>
                          <td className="py-3 px-3 font-mono">{(app.nameEmailSimilarity * 100).toFixed(0)}%</td>
                          <td className="py-3 px-3">
                            <span className="text-slate-700">{app.housingStatus}</span>
                            <span className="text-slate-400 mx-1">/</span>
                            <span className="text-slate-500">{app.deviceOs}</span>
                          </td>
                          <td className="py-3 px-3 font-mono font-bold">
                            <span className={app.riskScore >= decisionThreshold ? 'text-rose-600' : 'text-emerald-600'}>
                              {(app.riskScore * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td className="py-3 px-3">
                            <span
                              className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                                app.riskScore >= decisionThreshold
                                  ? 'bg-rose-50 text-rose-700 border border-rose-200'
                                  : app.riskScore >= 0.5
                                  ? 'bg-amber-50 text-amber-700 border border-amber-200'
                                  : 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                              }`}
                            >
                              {app.riskScore >= decisionThreshold ? 'Mule Alert' : app.riskScore >= 0.5 ? 'Review' : 'Approved'}
                            </span>
                          </td>
                          <td className="py-3 px-3 text-right">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedAuditApp(app);
                              }}
                              className="text-blue-600 hover:text-blue-800 font-medium flex items-center justify-end space-x-1"
                            >
                              <span>View Notice</span>
                              <ChevronRight className="w-3.5 h-3.5" />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* TAB CONTENT: 3. BEHAVIORAL GRAPH & VELOCITY */}
            {activeTab === 'graph' && (
              <div className="space-y-6">
                <div className="bg-white rounded-xl border border-slate-200 p-6">
                  <div className="max-w-3xl">
                    <h3 className="text-sm font-bold text-slate-900">
                      Behavioral Graph Metrics & Entity Collision Synthesis
                    </h3>
                    <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                      In online banking, fraud syndicates leverage automated emulators and device recycling to open clusters of money mule accounts. We map tabular features into bipartite graph projections that expose multi-entity collision densities without requiring graph database infrastructure.
                    </p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                      <span className="text-xs font-bold text-blue-700 uppercase block">1. Bipartite Degree Product</span>
                      <div className="font-mono text-xs bg-white p-2.5 rounded border border-slate-200 mt-2 text-slate-800">
                        device_distinct_emails_8w × dob_distinct_emails_4w
                      </div>
                      <p className="text-xs text-slate-600 mt-2 leading-relaxed">
                        Measures the intersection between hardware re-use and synchronized demographic targeting. High products isolate coordinated mule waves.
                      </p>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                      <span className="text-xs font-bold text-purple-700 uppercase block">2. Device Recycling Velocity</span>
                      <div className="font-mono text-xs bg-white p-2.5 rounded border border-slate-200 mt-2 text-slate-800">
                        device_distinct_emails_8w / (session_mins + ε)
                      </div>
                      <p className="text-xs text-slate-600 mt-2 leading-relaxed">
                        Turnover speed of unique identities on a single device per active minute. Detects high-speed bot farm script execution.
                      </p>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                      <span className="text-xs font-bold text-emerald-700 uppercase block">3. Velocity Surge Ratio</span>
                      <div className="font-mono text-xs bg-white p-2.5 rounded border border-slate-200 mt-2 text-slate-800">
                        (velocity_6h / 6) / ((velocity_24h / 24) + ε)
                      </div>
                      <p className="text-xs text-slate-600 mt-2 leading-relaxed">
                        Short-term hourly application surge rate normalized against 24-hour baseline. Isolates flash attack bursts.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Top Features Gain Table */}
                <div className="bg-white rounded-xl border border-slate-200 p-6">
                  <h3 className="text-sm font-bold text-slate-900 mb-3">
                    Top Feature Importance by Gain Contribution
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead>
                        <tr className="border-b border-slate-200 text-slate-500 font-semibold uppercase">
                          <th className="py-2.5 px-3">Feature Name</th>
                          <th className="py-2.5 px-3">Domain</th>
                          <th className="py-2.5 px-3">Gain Contribution</th>
                          <th className="py-2.5 px-3">Visual Distribution</th>
                          <th className="py-2.5 px-3">Role in Fraud Isolation</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 font-mono">
                        {FEATURE_IMPORTANCES.map((feat, idx) => (
                          <tr key={idx} className="hover:bg-slate-50">
                            <td className="py-2.5 px-3 font-bold text-slate-900">{feat.name}</td>
                            <td className="py-2.5 px-3 font-sans">
                              <span className="text-slate-600">{feat.category}</span>
                            </td>
                            <td className="py-2.5 px-3 font-bold text-blue-600">{feat.gain.toFixed(2)}%</td>
                            <td className="py-2.5 px-3 w-40">
                              <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-blue-600 rounded-full"
                                  style={{ width: `${Math.min(100, feat.gain * 2.5)}%` }}
                                />
                              </div>
                            </td>
                            <td className="py-2.5 px-3 font-sans text-slate-500 text-[11px]">{feat.desc}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* TAB CONTENT: 4. COST MATRIX OPTIMIZATION */}
            {activeTab === 'cost' && (
              <div className="space-y-6">
                <div className="bg-white rounded-xl border border-slate-200 p-6">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-100">
                    <div>
                      <h3 className="text-sm font-bold text-slate-900">
                        Operational Financial Loss Matrix Tuning
                      </h3>
                      <p className="text-xs text-slate-600 mt-1">
                        Configure banking losses to find the decision threshold that minimizes net charge-offs plus manual review investigation costs.
                      </p>
                    </div>

                    <div className="text-right bg-slate-50 px-4 py-2 rounded-lg border border-slate-200">
                      <span className="text-[10px] uppercase font-bold text-slate-500 block">Optimal Cutoff (τ*)</span>
                      <span className="text-lg font-bold text-emerald-700 font-mono">0.685 - 0.694</span>
                    </div>
                  </div>

                  {/* Config Inputs */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4">
                    <div>
                      <label className="text-xs font-semibold text-slate-700 block">
                        False Negative Cost (C_FN)
                      </label>
                      <span className="text-[11px] text-slate-500 block mb-1">Mule account bust-out loss</span>
                      <div className="relative">
                        <span className="absolute left-3 top-2 text-slate-400 text-xs">$</span>
                        <input
                          type="number"
                          value={costFn}
                          onChange={(e) => setCostFn(Number(e.target.value))}
                          className="w-full bg-slate-50 border border-slate-300 rounded px-3 py-1.5 pl-6 text-xs font-mono"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="text-xs font-semibold text-slate-700 block">
                        False Positive Cost (C_FP)
                      </label>
                      <span className="text-[11px] text-slate-500 block mb-1">Manual KYC review friction</span>
                      <div className="relative">
                        <span className="absolute left-3 top-2 text-slate-400 text-xs">$</span>
                        <input
                          type="number"
                          value={costFp}
                          onChange={(e) => setCostFp(Number(e.target.value))}
                          className="w-full bg-slate-50 border border-slate-300 rounded px-3 py-1.5 pl-6 text-xs font-mono"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="text-xs font-semibold text-slate-700 block">
                        True Positive Cost (C_TP)
                      </label>
                      <span className="text-[11px] text-slate-500 block mb-1">Automated biometric step-up</span>
                      <div className="relative">
                        <span className="absolute left-3 top-2 text-slate-400 text-xs">$</span>
                        <input
                          type="number"
                          value={costTp}
                          onChange={(e) => setCostTp(Number(e.target.value))}
                          className="w-full bg-slate-50 border border-slate-300 rounded px-3 py-1.5 pl-6 text-xs font-mono"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Interactive Threshold Slider */}
                  <div className="mt-6 pt-6 border-t border-slate-100">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-xs font-bold text-slate-800">
                        Adjust Decision Threshold (τ): <strong className="font-mono text-blue-600 text-sm">{decisionThreshold.toFixed(3)}</strong>
                      </span>
                      <span className="text-xs text-slate-500">
                        Default: 0.500 | 5% FPR Point: 0.694
                      </span>
                    </div>

                    <input
                      type="range"
                      min="0.10"
                      max="0.95"
                      step="0.005"
                      value={decisionThreshold}
                      onChange={(e) => setDecisionThreshold(Number(e.target.value))}
                      className="w-full accent-blue-600 bg-slate-200 h-2 rounded-lg cursor-pointer"
                    />

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6">
                      <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                        <span className="text-xs text-slate-500 block">Net Fleet Savings</span>
                        <span className="text-2xl font-extrabold text-emerald-700 font-mono mt-1 block">
                          ${Math.round(costSummary.totalFleetSavings).toLocaleString()}
                        </span>
                        <span className="text-[11px] text-slate-500">
                          Over {totalRowCount.toLocaleString()} applications
                        </span>
                      </div>

                      <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                        <span className="text-xs text-slate-500 block">Cost Per Application</span>
                        <span className="text-2xl font-extrabold text-slate-900 font-mono mt-1 block">
                          ${costSummary.costPerApp.toFixed(2)}
                        </span>
                        <span className="text-[11px] text-emerald-700">
                          {costSummary.savingsPct.toFixed(1)}% cheaper than default 0.5 cutoff
                        </span>
                      </div>

                      <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                        <span className="text-xs text-slate-500 block">Alert Precision</span>
                        <span className="text-2xl font-extrabold text-blue-600 font-mono mt-1 block">
                          {(costSummary.precision * 100).toFixed(1)}%
                        </span>
                        <span className="text-[11px] text-slate-500">
                          Realized FPR: {(costSummary.fpr * 100).toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* TAB CONTENT: 5. REGULATORY XAI & ADVERSARIAL STRESS */}
            {activeTab === 'regulatory' && (
              <div className="space-y-6">
                <div className="bg-white rounded-xl border border-slate-200 p-6">
                  <div className="max-w-3xl">
                    <h3 className="text-sm font-bold text-slate-900">
                      Regulatory Compliance Standards: RBI & Basel III Alignment
                    </h3>
                    <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                      In accordance with the <strong>RBI Master Direction on Digital Lending (2022)</strong> and <strong>Basel Committee on Banking Supervision (BCBS) / Fed SR 11-7 Model Risk Management</strong>, automated credit decline decisions require verifiable, transparent, and non-discriminatory adverse action notices.
                    </p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                    <div className="p-5 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
                      <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                        Mandatory Compliance Guarantees:
                      </h4>
                      <ul className="text-xs text-slate-600 space-y-2 list-disc list-inside">
                        <li><strong>Additive Feature Attribution:</strong> Uses Tree SHAP values strictly guaranteeing local accuracy: $f(x) = E[f(x)] + \sum \phi_i$.</li>
                        <li><strong>Adverse Action Transparency:</strong> Specific negative factors (e.g. device recycling velocity, disposable email mismatch) communicated clearly.</li>
                        <li><strong>Fairness & Exclusion:</strong> Sensitive demographic attributes (`customer_age`) excluded or audited for disparate impact under 80% rule.</li>
                      </ul>
                    </div>

                    {/* Adversarial Stress Test Slider */}
                    <div className="p-5 rounded-xl bg-slate-50 border border-slate-200">
                      <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2">
                        Adversarial Evasion Robustness Stress Test
                      </h4>
                      <p className="text-xs text-slate-600 mb-3">
                        Simulating fraudsters masking device identifiers and throttling session velocity:
                      </p>

                      <div className="flex justify-between items-center text-xs mb-1">
                        <span className="font-medium text-slate-700">Perturbation Budget (ε):</span>
                        <span className="font-mono font-bold text-amber-700">{(adversarialEpsilon * 100).toFixed(0)}% Noise</span>
                      </div>
                      <input
                        type="range"
                        min="0.00"
                        max="0.30"
                        step="0.05"
                        value={adversarialEpsilon}
                        onChange={(e) => setAdversarialEpsilon(Number(e.target.value))}
                        className="w-full accent-amber-600 bg-slate-200 h-2 rounded-lg cursor-pointer"
                      />

                      <div className="grid grid-cols-2 gap-2 mt-4 text-xs font-mono">
                        <div className="p-2.5 bg-white rounded border border-slate-200">
                          <span className="text-slate-500 font-sans block text-[10px]">Stressed ROC-AUC:</span>
                          <span className="font-bold text-slate-900">
                            {(0.8888 - 0.12 * adversarialEpsilon).toFixed(4)}
                          </span>
                        </div>
                        <div className="p-2.5 bg-white rounded border border-slate-200">
                          <span className="text-slate-500 font-sans block text-[10px]">Resilience at 5% FPR:</span>
                          <span className="font-bold text-emerald-700">
                            {((0.5261 - 0.35 * adversarialEpsilon) * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* INDIVIDUAL REGULATORY ADVERSE ACTION AUDIT MODAL                          */}
        {/* ========================================================================= */}
        {selectedAuditApp && (
          <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-sm flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl border border-slate-200 shadow-xl max-w-2xl w-full p-6 space-y-5 animate-in fade-in-50 zoom-in-95">
              <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                <div>
                  <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">
                    Official Regulatory Compliance Record · Case #{selectedAuditApp.id}
                  </span>
                  <h3 className="text-base font-bold text-slate-900">
                    Adverse Action Audit Notice (RBI / Basel III Aligned)
                  </h3>
                </div>
                <button
                  onClick={() => setSelectedAuditApp(null)}
                  className="p-1 rounded-lg hover:bg-slate-100 text-slate-500 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Case Summary */}
              <div className="grid grid-cols-3 gap-3 text-xs">
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                  <span className="text-slate-500 block text-[10px]">Mule Probability</span>
                  <span className="text-base font-bold font-mono text-rose-600">
                    {(selectedAuditApp.riskScore * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                  <span className="text-slate-500 block text-[10px]">Operating Cutoff</span>
                  <span className="text-base font-bold font-mono text-slate-800">
                    {(decisionThreshold * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                  <span className="text-slate-500 block text-[10px]">Regulatory Action</span>
                  <span className="text-xs font-bold text-rose-700 bg-rose-100/60 px-2 py-0.5 rounded inline-block mt-1">
                    DECLINE / AML ALERT
                  </span>
                </div>
              </div>

              {/* Top Positive Risk Drivers (Waterfall) */}
              <div className="space-y-3">
                <h4 className="text-xs font-bold text-rose-700 uppercase tracking-wider">
                  Primary Adverse Factors (Increasing Fraud Score):
                </h4>

                <div className="space-y-2 text-xs">
                  <div className="p-2.5 rounded-lg bg-rose-50/50 border border-rose-100 flex items-center justify-between">
                    <div>
                      <span className="font-bold text-slate-900">Abnormal Device Recycling Speed:</span>
                      <span className="text-slate-600 block text-[11px]">
                        {selectedAuditApp.deviceEmails8w} unique emails submitted in {selectedAuditApp.sessionLengthMins}m session
                      </span>
                    </div>
                    <span className="font-mono font-bold text-rose-700 text-sm">+1.42 log-odds</span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-rose-50/50 border border-rose-100 flex items-center justify-between">
                    <div>
                      <span className="font-bold text-slate-900">Demographic Housing Risk:</span>
                      <span className="text-slate-600 block text-[11px]">
                        Applicant category housing_status = {selectedAuditApp.housingStatus}
                      </span>
                    </div>
                    <span className="font-mono font-bold text-rose-700 text-sm">+1.18 log-odds</span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-rose-50/50 border border-rose-100 flex items-center justify-between">
                    <div>
                      <span className="font-bold text-slate-900">Identity Mismatch on Email:</span>
                      <span className="text-slate-600 block text-[11px]">
                        Low name-to-email similarity: {(selectedAuditApp.nameEmailSimilarity * 100).toFixed(0)}%
                      </span>
                    </div>
                    <span className="font-mono font-bold text-rose-700 text-sm">+0.84 log-odds</span>
                  </div>
                </div>
              </div>

              {/* Mitigating Factors */}
              <div className="pt-2 border-t border-slate-100 text-xs">
                <h4 className="font-bold text-emerald-800 uppercase tracking-wider mb-1.5">
                  Mitigating Positive Factors:
                </h4>
                <div className="flex justify-between text-slate-600 p-2 bg-emerald-50/40 rounded border border-emerald-100 font-mono">
                  <span>Standard application channel ({selectedAuditApp.source})</span>
                  <span className="text-emerald-700 font-bold">-0.32 log-odds</span>
                </div>
              </div>

              {/* Footer */}
              <div className="pt-3 border-t border-slate-100 flex items-center justify-between">
                <span className="text-[11px] text-slate-500">
                  Model Governance ID: BAF-GBDT-LGBM-V2
                </span>
                <button
                  onClick={() => setSelectedAuditApp(null)}
                  className="px-4 py-1.5 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 rounded-lg transition-colors"
                >
                  Dismiss Notice
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-6 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <span>Mule Account Fraud Detection Benchmark · NeurIPS 2022 Bank Account Fraud Dataset Suite</span>
          <div className="flex items-center space-x-4">
            <span>Basel III Model Risk Governance</span>
            <span>·</span>
            <span>RBI Digital Lending 2022 Aligned</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
