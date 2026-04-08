"use client";

import { useState } from "react";
import { signIn } from "@/lib/auth-client";
import Link from "next/link";

export default function SignInPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleEmailSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await signIn.email({ email, password, callbackURL: "/dashboard" });
      if (res.error) setError(res.error.message || "Sign in failed");
    } catch {
      setError("An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    await signIn.social({ provider: "google", callbackURL: "/dashboard" });
  };

  return (
    <div className="auth-page">
      {/* Animated background orbs */}
      <div className="auth-bg-orbs">
        <div className="auth-orb auth-orb-1" />
        <div className="auth-orb auth-orb-2" />
        <div className="auth-orb auth-orb-3" />
      </div>

      <div className="auth-card">
        {/* Glow border */}
        <div className="auth-card-glow" />

        <div className="auth-card-inner">
          {/* Header */}
          <div className="auth-header">
            <div className="auth-logo-mark">
              <span>🗺️</span>
            </div>
            <h1>Welcome back</h1>
            <p>Sign in to your ATLAS account</p>
          </div>

          {/* Google */}
          <button type="button" className="auth-social-btn" onClick={handleGoogleSignIn}>
            <svg width="18" height="18" viewBox="0 0 48 48">
              <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
              <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
              <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
              <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
            </svg>
            Continue with Google
          </button>

          <div className="auth-divider">
            <div className="auth-divider-line" />
            <span>or</span>
            <div className="auth-divider-line" />
          </div>

          {/* Form */}
          <form onSubmit={handleEmailSignIn}>
            <div className="auth-field-group">
              <label htmlFor="email">Email</label>
              <div className="auth-input-wrap">
                <span className="auth-input-icon">✉️</span>
                <input
                  id="email" type="email" value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com" required
                />
              </div>
            </div>

            <div className="auth-field-group">
              <label htmlFor="password">Password</label>
              <div className="auth-input-wrap">
                <span className="auth-input-icon">🔒</span>
                <input
                  id="password" type="password" value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••" required
                />
              </div>
            </div>

            {error && (
              <div className="auth-error-toast">
                <span>⚠️</span> {error}
              </div>
            )}

            <button type="submit" className="auth-primary-btn" disabled={loading}>
              {loading ? (
                <span className="auth-btn-loading">
                  <span className="auth-spinner" /> Signing in…
                </span>
              ) : (
                "Sign In →"
              )}
            </button>
          </form>

          <p className="auth-footer-text">
            Don&apos;t have an account?{" "}
            <Link href="/sign-up">Create one</Link>
          </p>
        </div>
      </div>
    </div>
  );
}
