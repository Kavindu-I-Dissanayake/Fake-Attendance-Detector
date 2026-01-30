// app/(tabs)/live/enroll.tsx
import React, { useRef, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  Alert,
  Image,
  ActivityIndicator,
  ScrollView,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as ImageManipulator from "expo-image-manipulator";
import { BASE_URL } from "../../../config";

export default function EnrollScreen() {
  const cameraRef = useRef<CameraView>(null);

  const [permission, requestPermission] = useCameraPermissions();

  const [studentId, setStudentId] = useState("");
  const [name, setName] = useState("");

  const [photos, setPhotos] = useState<string[]>([]);
  const [cameraOn, setCameraOn] = useState(false);

  const [loading, setLoading] = useState(false);

  // 1) Permission UI
  if (!permission) return <View style={styles.container} />;
  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>Enroll Student</Text>
        <Text style={styles.sub}>Camera permission is required.</Text>

        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Allow Camera</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // 2) Capture a photo (front camera)
  const capturePhoto = async () => {
    try {
      if (!cameraRef.current) return;

      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.7,
        skipProcessing: true,
      });

      // compress + resize for faster upload (still clear enough for face)
      const processed = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: 900 } }],
        { compress: 0.7, format: ImageManipulator.SaveFormat.JPEG }
      );

      setPhotos((prev) => [...prev, processed.uri]);
    } catch (e) {
      Alert.alert("Error", "Failed to capture photo.");
    }
  };

  const resetPhotos = () => {
    setPhotos([]);
    setCameraOn(false);
  };

  // 3) Submit to backend
  const submitEnroll = async () => {
    if (!studentId.trim() || !name.trim()) {
      Alert.alert("Missing fields", "Please enter studentId and name.");
      return;
    }
    if (photos.length !== 3) {
      Alert.alert("Need 3 photos", "Capture exactly 3 face photos.");
      return;
    }

    setLoading(true);
    try {
      const form = new FormData();
      form.append("studentId", studentId.trim());
      form.append("name", name.trim());
      form.append("className", "Group-Project");

      photos.forEach((uri, idx) => {
        form.append("images", {
          uri,
          name: `face_${idx + 1}.jpg`,
          type: "image/jpeg",
        } as any);
      });

      const res = await fetch(`${BASE_URL}/face/enroll`, {
        method: "POST",
        body: form,
        // NOTE: don't manually set Content-Type for FormData in RN fetch
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Enroll failed");
      }

      Alert.alert("Success", "Student enrolled successfully!");
      setStudentId("");
      setName("");
      resetPhotos();
    } catch (e: any) {
      Alert.alert("Enroll failed", e?.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  // 4) UI
  return (
    <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
      <Text style={styles.title}>Enroll Student</Text>
      <Text style={styles.sub}>
        Enter details, then capture 3 front-face photos.
      </Text>

      <View style={styles.card}>
        <TextInput
          placeholder="studentId"
          placeholderTextColor="#777"
          style={styles.input}
          value={studentId}
          onChangeText={setStudentId}
          autoCapitalize="none"
        />
        <TextInput
          placeholder="name"
          placeholderTextColor="#777"
          style={styles.input}
          value={name}
          onChangeText={setName}
        />

        <Text style={styles.note}>
          Class: <Text style={styles.bold}>Group-Project</Text>
        </Text>
      </View>

      {/* Camera area */}
      {cameraOn && photos.length < 3 ? (
        <View style={styles.cameraWrap}>
          <CameraView ref={cameraRef} style={styles.camera} facing="front" />
          <Text style={styles.counter}>Photo {photos.length + 1} / 3</Text>

          <TouchableOpacity style={styles.captureBtn} onPress={capturePhoto} activeOpacity={0.85}>
            <Text style={styles.captureText}>Capture</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.secondaryBtn} onPress={() => setCameraOn(false)}>
            <Text style={styles.secondaryText}>Close Camera</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <TouchableOpacity
          style={styles.button}
          onPress={() => setCameraOn(true)}
          disabled={photos.length === 3}
        >
          <Text style={styles.buttonText}>
            {photos.length === 3 ? "Captured 3 Photos" : "Open Front Camera"}
          </Text>
        </TouchableOpacity>
      )}

      {/* Photo previews */}
      {photos.length > 0 && (
        <View style={styles.previewRow}>
          {photos.map((uri, i) => (
            <Image key={i} source={{ uri }} style={styles.previewImg} />
          ))}
        </View>
      )}

      {/* Actions */}
      <TouchableOpacity
        style={[styles.button, { opacity: loading ? 0.7 : 1 }]}
        onPress={submitEnroll}
        disabled={loading}
      >
        {loading ? <ActivityIndicator /> : <Text style={styles.buttonText}>Submit Enrollment</Text>}
      </TouchableOpacity>

      <TouchableOpacity style={styles.secondaryBtn} onPress={resetPhotos}>
        <Text style={styles.secondaryText}>Reset Photos</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: "#111",
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  title: {
    color: "#fff",
    fontSize: 26,
    fontWeight: "bold",
    textAlign: "center",
  },
  sub: {
    color: "#aaa",
    fontSize: 16,
    marginTop: 14,
    textAlign: "center",
    marginBottom: 18,
  },
  card: {
    width: "100%",
    maxWidth: 420,
    borderWidth: 1,
    borderColor: "#333",
    borderRadius: 12,
    padding: 14,
    backgroundColor: "#000",
    marginBottom: 16,
  },
  input: {
    width: "100%",
    borderWidth: 1,
    borderColor: "#333",
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: "#fff",
    marginBottom: 10,
    backgroundColor: "#111",
  },
  note: { color: "#888", fontSize: 12, textAlign: "center", marginTop: 6 },
  bold: { color: "#fff", fontWeight: "bold" },

  button: {
    width: "100%",
    maxWidth: 320,
    paddingVertical: 14,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#333",
    backgroundColor: "#000",
    marginBottom: 14,
    alignItems: "center",
  },
  buttonText: { color: "#fff", fontSize: 16, fontWeight: "600" },

  cameraWrap: {
    width: "100%",
    maxWidth: 420,
    borderWidth: 1,
    borderColor: "#333",
    borderRadius: 12,
    overflow: "hidden",
    backgroundColor: "#000",
    marginBottom: 14,
    alignItems: "center",
  },
  camera: { width: "100%", height: 320 },
  counter: { color: "#aaa", marginTop: 10, marginBottom: 10 },

  captureBtn: {
    width: "90%",
    paddingVertical: 12,
    borderRadius: 10,
    backgroundColor: "#00A8FF",
    alignItems: "center",
    marginBottom: 10,
  },
  captureText: { color: "#000", fontWeight: "800", fontSize: 16 },

  secondaryBtn: {
    paddingVertical: 10,
    paddingHorizontal: 14,
    marginBottom: 12,
  },
  secondaryText: { color: "#aaa", textDecorationLine: "underline" },

  previewRow: {
    flexDirection: "row",
    gap: 10,
    marginBottom: 14,
    justifyContent: "center",
    flexWrap: "wrap",
  },
  previewImg: {
    width: 90,
    height: 90,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#333",
  },
});
