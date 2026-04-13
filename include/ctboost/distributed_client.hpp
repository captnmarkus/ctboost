#pragma once

#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace ctboost {

struct DistributedTcpRoot {
  bool is_tcp{false};
  std::string host;
  std::uint16_t port{0};
};

inline bool DistributedRootUsesTcp(const std::string& root) {
  return root.rfind("tcp://", 0) == 0;
}

inline DistributedTcpRoot ParseDistributedTcpRoot(const std::string& root) {
  if (!DistributedRootUsesTcp(root)) {
    return DistributedTcpRoot{};
  }
  const std::string endpoint_with_path = root.substr(6);
  const std::size_t slash = endpoint_with_path.find('/');
  const std::string endpoint =
      slash == std::string::npos ? endpoint_with_path : endpoint_with_path.substr(0, slash);
  const std::size_t colon = endpoint.rfind(':');
  if (colon == std::string::npos || colon == 0 || colon + 1 >= endpoint.size()) {
    throw std::invalid_argument("distributed tcp root must be formatted like tcp://host:port");
  }
  const int parsed_port = std::stoi(endpoint.substr(colon + 1));
  if (parsed_port <= 0 || parsed_port > 65535) {
    throw std::invalid_argument("distributed tcp port must be in [1, 65535]");
  }
  return DistributedTcpRoot{
      true, endpoint.substr(0, colon), static_cast<std::uint16_t>(parsed_port)};
}

namespace distributed_client_detail {

#ifdef _WIN32
using SocketHandle = SOCKET;
constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
#else
using SocketHandle = int;
constexpr SocketHandle kInvalidSocket = -1;
#endif

struct SocketGuard {
  SocketHandle socket{kInvalidSocket};
  ~SocketGuard() {
    if (socket != kInvalidSocket) {
#ifdef _WIN32
      closesocket(socket);
#else
      close(socket);
#endif
    }
  }
};

#ifdef _WIN32
inline void EnsureSocketLibraryInitialized() {
  static bool initialized = []() {
    WSADATA data;
    const int result = WSAStartup(MAKEWORD(2, 2), &data);
    if (result != 0) {
      throw std::runtime_error("WSAStartup failed for distributed tcp client");
    }
    return true;
  }();
  (void)initialized;
}
#else
inline void EnsureSocketLibraryInitialized() {}
#endif

inline void SendAll(SocketHandle socket, const char* data, std::size_t size) {
  std::size_t sent = 0;
  while (sent < size) {
#ifdef _WIN32
    const int result =
        send(socket, data + static_cast<std::ptrdiff_t>(sent), static_cast<int>(size - sent), 0);
#else
    const ssize_t result =
        send(socket, data + static_cast<std::ptrdiff_t>(sent), size - sent, 0);
#endif
    if (result <= 0) {
      throw std::runtime_error("distributed tcp send failed");
    }
    sent += static_cast<std::size_t>(result);
  }
}

inline void ReceiveAll(SocketHandle socket, char* data, std::size_t size) {
  std::size_t received = 0;
  while (received < size) {
#ifdef _WIN32
    const int result = recv(
        socket, data + static_cast<std::ptrdiff_t>(received), static_cast<int>(size - received), 0);
#else
    const ssize_t result =
        recv(socket, data + static_cast<std::ptrdiff_t>(received), size - received, 0);
#endif
    if (result <= 0) {
      throw std::runtime_error("distributed tcp receive failed");
    }
    received += static_cast<std::size_t>(result);
  }
}

inline std::string ReceiveLine(SocketHandle socket) {
  std::string line;
  while (true) {
    char value = '\0';
    ReceiveAll(socket, &value, 1);
    if (value == '\n') {
      return line;
    }
    line.push_back(value);
  }
}

inline SocketHandle ConnectSocketWithRetry(const DistributedTcpRoot& root, double timeout_seconds) {
  EnsureSocketLibraryInitialized();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_seconds);

  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  const std::string port = std::to_string(root.port);

  while (true) {
    addrinfo* results = nullptr;
    const int resolve_result = getaddrinfo(root.host.c_str(), port.c_str(), &hints, &results);
    if (resolve_result != 0) {
      if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error("distributed tcp getaddrinfo failed");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      continue;
    }

    SocketHandle connected = kInvalidSocket;
    for (addrinfo* current = results; current != nullptr; current = current->ai_next) {
      SocketHandle candidate =
          socket(current->ai_family, current->ai_socktype, current->ai_protocol);
      if (candidate == kInvalidSocket) {
        continue;
      }
      const int connect_result = connect(candidate, current->ai_addr, static_cast<int>(current->ai_addrlen));
      if (connect_result == 0) {
        connected = candidate;
        break;
      }
#ifdef _WIN32
      closesocket(candidate);
#else
      close(candidate);
#endif
    }
    freeaddrinfo(results);

    if (connected != kInvalidSocket) {
      return connected;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error("timed out connecting to distributed tcp coordinator");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

}  // namespace distributed_client_detail

inline std::vector<std::uint8_t> DistributedTcpRequest(const std::string& root,
                                                       double timeout_seconds,
                                                       const std::string& op,
                                                       const std::string& key,
                                                       int rank,
                                                       int world_size,
                                                       const std::vector<std::uint8_t>& payload) {
  const DistributedTcpRoot parsed = ParseDistributedTcpRoot(root);
  if (!parsed.is_tcp) {
    throw std::invalid_argument("distributed tcp request requires a tcp://host:port root");
  }

  distributed_client_detail::SocketGuard socket{
      distributed_client_detail::ConnectSocketWithRetry(parsed, timeout_seconds)};
  const std::string header = op + "\t" + key + "\t" + std::to_string(rank) + "\t" +
                             std::to_string(world_size) + "\t" +
                             std::to_string(payload.size()) + "\n";
  distributed_client_detail::SendAll(socket.socket, header.data(), header.size());
  if (!payload.empty()) {
    distributed_client_detail::SendAll(
        socket.socket, reinterpret_cast<const char*>(payload.data()), payload.size());
  }
  const std::string response_line = distributed_client_detail::ReceiveLine(socket.socket);
  const std::size_t tab = response_line.find('\t');
  if (tab == std::string::npos) {
    throw std::runtime_error("invalid distributed coordinator response");
  }
  const std::string status = response_line.substr(0, tab);
  const std::string remainder = response_line.substr(tab + 1);
  if (status != "ok") {
    throw std::runtime_error(remainder);
  }
  const std::size_t response_size = static_cast<std::size_t>(std::stoull(remainder));
  std::vector<std::uint8_t> response(response_size, 0U);
  if (response_size != 0U) {
    distributed_client_detail::ReceiveAll(
        socket.socket, reinterpret_cast<char*>(response.data()), response.size());
  }
  return response;
}

}  // namespace ctboost
