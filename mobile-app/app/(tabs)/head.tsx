import { useState, useEffect } from "react";
import {
  View,
  Text,
  Pressable,
  StyleSheet,
  ActivityIndicator,
  Alert,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import * as ImagePicker from "expo-image-picker";
import { useKeepAwake } from "expo-keep-awake";
import { useAttendance } from "../../context/AttendanceContext";

export default function HeadScreen() {
  useKeepAwake();
  const { setHeadCount } = useAttendance();

  const [mediaPermission, setMediaPermission] =
    useState<ImagePicker.PermissionStatus | null>(null);

  const [isUploading, setIsUploading] = useState(false);
  const [status, setStatus] = useState(
    "Welcome! Select a video to analyze."
  );

  useEffect(() => {
    (async () => {
      const { status } =
        await ImagePicker.requestMediaLibraryPermissionsAsync();
      setMediaPermission(status);
    })();
  }, []);

  const selectAndUploadVideo = async () => {
    if (mediaPermission !== "granted") {
      Alert.alert("Permission", "Media access required.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: "videos",
      quality: 1,
    });

    if (!result.canceled && result.assets.length > 0) {
      await uploadVideo(result.assets[0].uri);
    }
  };

  const uploadVideo = async (uri: string) => {
    const API_URL = "http://10.14.157.251:8000/upload/video";
    setIsUploading(true);
    setStatus("Uploading video...");

    try {
      const formData = new FormData();
      formData.append("video_file", {
        uri,
        name: "video.mp4",
        type: "video/mp4",
      } as any);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      setHeadCount(Number(data.head_count));
      setStatus("Head count updated!");
    } catch (err) {
      console.error(err);
      setStatus("Failed to upload.");
    }

    setIsUploading(false);
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.title}>Head Detection</Text>

        <Text style={styles.status}>{status}</Text>

        <Pressable
          style={[styles.button, isUploading && styles.disabled]}
          onPress={selectAndUploadVideo}
        >
          {isUploading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>Select & Upload Video</Text>
          )}
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#1e1e1e" },
  content: { flex: 1, padding: 20 },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    textAlign: "center",
    color: "white",
  },
  status: { color: "#ccc", textAlign: "center", marginVertical: 20 },
  button: {
    backgroundColor: "#007AFF",
    padding: 16,
    borderRadius: 10,
    alignItems: "center",
  },
  disabled: { backgroundColor: "#555" },
  buttonText: { color: "white", fontSize: 18, fontWeight: "600" },
});
