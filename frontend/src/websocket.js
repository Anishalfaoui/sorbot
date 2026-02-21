import { Client } from '@stomp/stompjs';
import SockJS from 'sockjs-client';

let client = null;
const subscribers = {};

export function connectWebSocket(onConnect) {
  if (client && client.connected) return;

  client = new Client({
    webSocketFactory: () => new SockJS('http://localhost:8081/ws'),
    reconnectDelay: 5000,
    heartbeatIncoming: 10000,
    heartbeatOutgoing: 10000,
    debug: () => {},
  });

  client.onConnect = () => {
    console.log('[WS] Connected');
    if (onConnect) onConnect();

    // Re-subscribe existing topics on reconnect
    Object.entries(subscribers).forEach(([topic, callbacks]) => {
      callbacks.forEach((cb) => {
        client.subscribe(topic, (message) => {
          try {
            cb(JSON.parse(message.body));
          } catch (e) {
            cb(message.body);
          }
        });
      });
    });
  };

  client.onDisconnect = () => console.log('[WS] Disconnected');
  client.onStompError = (frame) => console.error('[WS] Error:', frame.headers?.message);

  client.activate();
}

export function subscribe(topic, callback) {
  if (!subscribers[topic]) subscribers[topic] = [];
  subscribers[topic].push(callback);

  if (client && client.connected) {
    client.subscribe(topic, (message) => {
      try {
        callback(JSON.parse(message.body));
      } catch (e) {
        callback(message.body);
      }
    });
  }
}

export function isConnected() {
  return client && client.connected;
}

export function disconnectWebSocket() {
  if (client) {
    client.deactivate();
    client = null;
  }
}
