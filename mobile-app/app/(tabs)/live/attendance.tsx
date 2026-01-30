// app/(tabs)/live/attendance.tsx
import React, { useRef, useState } from "react";
import { useRouter } from "expo-router";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as ImageManipulator from "expo-image-manipulator";
import { BASE_URL } from "../../../config";

type VerifyStatus = "idle" | "unknown" | "verified" | "already_marked" | "error";

export default function AttendanceScreen() {
  const cameraRef = useRef<CameraView>(null);
  const router = useRouter();

  const [permission, requestPermission] = useCameraPermissions();

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loadingSession, setLoadingSession] = useState(false);
  const [scanning, setScanning] = useState(false);

  const [status, setStatus] = useState<VerifyStatus>("idle");
  const [message, setMessage] = useState<string>("Start a session to begin.");
  const [lastStudent, setLastStudent] = useState<string>("");

  const pickBoxColor = () => {
    if (status === "verified" || status === "already_marked") return "#00ff66"; // green
    if (status === "unknown" || status === "error") return "#ff2a2a"; // red
    return "#888"; // idle
  };

  const startSession = async () => {
    setLoadingSession(true);
    try {
      const form = new FormData();
      form.append("className", "Group-Project");

      const res = await fetch(`${BASE_URL}/face/session/start`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail || "Failed to start session");
      }

      const data = await res.json();
      setSessionId(data.sessionId);
      setStatus("idle");
      setLastStudent("");
      setMessage("Session started. Tap 'Scan Face' for each student.");
    } catch (e: any) {
      Alert.alert("Error", e?.message || "Failed to start session");
    } finally {
      setLoadingSession(false);
    }
  };

  const captureAndVerifyOnce = async () => {
    if (!sessionId) {
      Alert.alert("Session not started", "Tap 'Start Session' first.");
      return;
    }
    if (!cameraRef.current) {
      Alert.alert("Camera not ready", "Try again in a moment.");
      return;
    }

    setScanning(true);
    setMessage("Scanning...");

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.6,
        skipProcessing: true,
      });

      const processed = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: 900 } }],
        { compress: 0.7, format: ImageManipulator.SaveFormat.JPEG }
      );

      const form = new FormData();
      form.append("sessionId", sessionId);
      form.append("image", {
        uri: processed.uri,
        name: "scan.jpg",
        type: "image/jpeg",
      } as any);

      const res = await fetch(`${BASE_URL}/face/session/verify`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setStatus("error");
        setLastStudent("");
        setMessage(err?.detail || "Verify failed");
        return;
      }

      const data = await res.json();

      if (data.status === "verified" || data.status === "already_marked") {
        const sid = data?.student?.studentId || "";
        const sname = data?.student?.name || "";
        setStatus(data.status);
        setLastStudent(sname ? `${sid} - ${sname}` : sid);
        setMessage(
          data.status === "already_marked"
            ? "Already marked present"
            : "Verified and marked present"
        );
      } else {
        setStatus("unknown");
        setLastStudent("");
        setMessage(data?.message || "Face not recognized");
      }
    } catch (e: any) {
      setStatus("error");
      setLastStudent("");
      setMessage("Camera/Network error");
    } finally {
      setScanning(false);
    }
  };

  const goToReport = () => {
    if (!sessionId) {
      Alert.alert("No session", "Start a session first.");
      return;
    }
    router.push(`/live/report?sessionId=${sessionId}`);
  };

  // --- Permission UI ---
  if (!permission) return <View style={styles.container} />;

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>Take Attendance</Text>
        <Text style={styles.sub}>Camera permission is required.</Text>

        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Allow Camera</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Take Attendance</Text>
      <Text style={styles.sub}>
        Session:{" "}
        <Text style={styles.bold}>
          {sessionId ? sessionId.slice(0, 8) + "..." : "Not started"}
        </Text>
      </Text>

      <View style={styles.cameraWrap}>
        <CameraView ref={cameraRef} style={styles.camera} facing="front" />

        {/* Prototype bounding box (UI feedback) */}
        <View style={[styles.box, { borderColor: pickBoxColor() }]} />

        <View style={styles.overlayTextWrap}>
          <Text style={styles.overlayText}>{message}</Text>
          {!!lastStudent && (
            <Text style={styles.overlaySub}>Student: {lastStudent}</Text>
          )}
        </View>
      </View>

      {/* Buttons */}
      {!sessionId ? (
        <TouchableOpacity
          style={[styles.button, { opacity: loadingSession ? 0.7 : 1 }]}
          onPress={startSession}
          disabled={loadingSession}
        >
          {loadingSession ? (
            <ActivityIndicator />
          ) : (
            <Text style={styles.buttonText}>Start Session</Text>
          )}
        </TouchableOpacity>
      ) : (
        <>
          <TouchableOpacity
            style={[styles.button, { opacity: scanning ? 0.7 : 1 }]}
            onPress={captureAndVerifyOnce}
            disabled={scanning}
          >
            {scanning ? (
              <ActivityIndicator />
            ) : (
              <Text style={styles.buttonText}>Scan Face</Text>
            )}
          </TouchableOpacity>

          <TouchableOpacity style={styles.secondaryBtn} onPress={goToReport}>
            <Text style={styles.secondaryText}>View Report</Text>
          </TouchableOpacity>
        </>
      )}

      {/* Small helper */}
      <Text style={styles.helper}>
        Tip: Pass the phone to each student and tap{" "}
        <Text style={styles.bold}>"Scan Face"</Text>.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#111",
    justifyContent: "flex-start",
    alignItems: "center",
    padding: 20,
    paddingTop: 30,
  },
  title: {
    color: "#fff",
    fontSize: 26,
    fontWeight: "bold",
    textAlign: "center",
  },
  sub: {
    color: "#aaa",
    fontSize: 14,
    marginTop: 10,
    textAlign: "center",
    marginBottom: 14,
  },
  bold: { color: "#fff", fontWeight: "bold" },

  cameraWrap: {
    width: "100%",
    maxWidth: 420,
    height: 360,
    borderWidth: 1,
    borderColor: "#333",
    borderRadius: 12,
    overflow: "hidden",
    backgroundColor: "#000",
    marginBottom: 16,
  },
  camera: { width: "100%", height: "100%" },

  // centered box overlay (prototype)
  box: {
    position: "absolute",
    left: "18%",
    top: "18%",
    width: "64%",
    height: "64%",
    borderWidth: 3,
    borderRadius: 10,
  },

  overlayTextWrap: {
    position: "absolute",
    bottom: 10,
    left: 10,
    right: 10,
    backgroundColor: "rgba(0,0,0,0.55)",
    padding: 10,
    borderRadius: 10,
  },
  overlayText: { color: "#fff", fontSize: 14, fontWeight: "700" },
  overlaySub: { color: "#ddd", fontSize: 12, marginTop: 4 },

  button: {
    width: "100%",
    maxWidth: 320,
    paddingVertical: 14,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#333",
    backgroundColor: "#000",
    marginBottom: 10,
    alignItems: "center",
  },
  buttonText: { color: "#fff", fontSize: 16, fontWeight: "600" },

  secondaryBtn: {
    paddingVertical: 10,
    paddingHorizontal: 14,
    marginBottom: 4,
  },
  secondaryText: {
    color: "#aaa",
    textDecorationLine: "underline",
    fontSize: 14,
  },

  helper: {
    color: "#888",
    fontSize: 12,
    textAlign: "center",
    marginTop: 4,
  },
});
