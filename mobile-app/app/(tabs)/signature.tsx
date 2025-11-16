import { useState, useEffect } from "react";
import {
  View,
  Text,
  Pressable,
  StyleSheet,
  ActivityIndicator,
  Alert,
  Image,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import * as ImagePicker from "expo-image-picker";
import { useKeepAwake } from "expo-keep-awake";
import { useAttendance } from "../../context/AttendanceContext";

export default function SignatureScreen() {
  useKeepAwake();

  const { setSignatureCount, setAbsentees } = useAttendance();

  const [mediaPermission, setMediaPermission] =
    useState<ImagePicker.PermissionStatus | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [status, setStatus] = useState(
    "Upload a signature sheet to count signatures."
  );
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  // Request media permissions
  useEffect(() => {
    (async () => {
      const { status } =
        await ImagePicker.requestMediaLibraryPermissionsAsync();
      setMediaPermission(status);
    })();
  }, []);

  // Select image
  const selectAndUploadImage = async () => {
    if (mediaPermission !== "granted") {
      Alert.alert("Permission required", "Please allow gallery access.");
      return;
    }

    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: "images",
        quality: 1,
      });

      if (!result.canceled && result.assets.length > 0) {
        const uri = result.assets[0].uri;
        setSelectedImage(uri);
        uploadSheet(uri);
      }
    } catch (err) {
      console.log("Image selection error:", err);
      Alert.alert("Error", "Failed to pick image.");
    }
  };

  // Upload to backend
  const uploadSheet = async (uri: string) => {
    setIsUploading(true);
    setStatus("Processing signature sheet...");

    try {
      const filename = uri.split("/").pop() || "sheet.jpg";
      const match = /\.(\w+)$/.exec(filename);
      const type = match ? `image/${match[1]}` : "image/jpeg";

      const formData = new FormData();
      formData.append("sheet_file", {
        uri,
        name: filename,
        type,
      } as any);

      const API_URL = "http://10.14.157.251:8000/upload/signsheet";

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
        headers: {
          Accept: "application/json",
        },
      });

      const data = await response.json();

      console.log("SIGNATURE API RESPONSE → ", data);

      // Update global context
      setSignatureCount(data.present_count ?? 0);
      setAbsentees(data.absentees ?? []);

      setStatus(`Found ${data.present_count} signatures.`);
    } catch (err) {
      console.log("Upload error:", err);
      Alert.alert("Error", "Failed to process the signature sheet.");
      setStatus("Upload failed.");
    } finally {
      setIsUploading(false);
    }
  };

  if (mediaPermission !== "granted") {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={{ color: "#fff" }}>
          Please allow media access in settings.
        </Text>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Signature Sheet Processing</Text>

      <View style={styles.imageContainer}>
        {selectedImage ? (
          <Image
            source={{ uri: selectedImage }}
            style={styles.image}
            resizeMode="contain"
          />
        ) : (
          <Text style={styles.placeholder}>No image selected</Text>
        )}

        {isUploading && (
          <View style={styles.loadingOverlay}>
            <ActivityIndicator size="large" color="#007AFF" />
          </View>
        )}
      </View>

      <Pressable
        style={[styles.button, isUploading && styles.buttonDisabled]}
        onPress={selectAndUploadImage}
        disabled={isUploading}
      >
        <Text style={styles.buttonText}>
          {isUploading ? "Processing..." : "Select Signature Sheet"}
        </Text>
      </Pressable>

      <Text style={styles.status}>{status}</Text>
    </SafeAreaView>
  );
}

/* ------------------------- STYLES ------------------------- */

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#1E1E1E",
    paddingHorizontal: 20,
    paddingTop: 20,
  },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#1E1E1E",
  },
  title: {
    fontSize: 22,
    fontWeight: "bold",
    color: "white",
    textAlign: "center",
    marginBottom: 15,
  },
  imageContainer: {
    width: "100%",
    height: 280,
    backgroundColor: "#2A2A2A",
    borderRadius: 10,
    overflow: "hidden",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 15,
  },
  placeholder: {
    color: "#aaa",
  },
  image: {
    width: "100%",
    height: "100%",
  },
  loadingOverlay: {
    position: "absolute",
    justifyContent: "center",
    alignItems: "center",
    width: "100%",
    height: "100%",
    backgroundColor: "rgba(0,0,0,0.4)",
  },
  button: {
    backgroundColor: "#007AFF",
    padding: 14,
    borderRadius: 8,
    marginTop: 5,
  },
  buttonDisabled: {
    backgroundColor: "#4A90E2",
  },
  buttonText: {
    color: "white",
    fontSize: 16,
    textAlign: "center",
    fontWeight: "600",
  },
  status: {
    textAlign: "center",
    color: "#ccc",
    marginTop: 10,
    fontSize: 14,
  },
});
