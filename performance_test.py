#!/usr/bin/env python3
"""
Performance test script for the optimized Turkish lemmatizer
Tests batch processing, caching, and parallel execution
"""

import asyncio
import aiohttp
import time
import json
from typing import List

# Test data - realistic Turkish n-grams
TEST_DATA = [
    "merkezi sistem yönetimi",
    "idari yaptırım uygulaması", 
    "kanuni faiz hesaplaması",
    "hukuki süreç takibi",
    "mali sorumluluk sigortası",
    "ticari işlem kayıtları",
    "resmi belge düzenleme",
    "kamu hizmet sunumu",
    "sosyal güvenlik sistemi",
    "eğitim öğretim faaliyetleri",
    "sağlık hizmetleri yönetimi",
    "çevre koruma önlemleri",
    "teknoloji geliştirme projesi",
    "araştırma geliştirme faaliyeti",
    "kalite kontrol süreci",
    "güvenlik önlemleri uygulaması",
    "performans değerlendirme sistemi",
    "stratejik planlama süreci",
    "risk yönetimi uygulaması",
    "sürekli iyileştirme faaliyeti"
]

# Generate large test dataset
def generate_large_dataset(base_data: List[str], multiplier: int = 100) -> List[str]:
    """Generate a large dataset by repeating base data"""
    large_dataset = []
    for i in range(multiplier):
        for item in base_data:
            # Add some variation to test caching
            variation = f" {i}" if i % 10 == 0 else ""
            large_dataset.append(item + variation)
    return large_dataset

async def test_performance(session: aiohttp.ClientSession, texts: List[str], test_name: str):
    """Test performance for a given dataset"""
    print(f"\n🧪 {test_name}")
    print(f"📊 Testing with {len(texts)} texts...")
    
    payload = {"texts": texts, "return_details": True}
    
    start_time = time.time()
    
    try:
        async with session.post('http://localhost:8077/lemmatize', json=payload) as response:
            if response.status == 200:
                result = await response.json()
                processing_time = (time.time() - start_time) * 1000
                
                # Extract performance metrics
                perf_data = result.get('performance', {})
                api_time = perf_data.get('processing_time_ms', 0)
                cache_info = perf_data.get('cache_hits')
                
                print(f"✅ Success!")
                print(f"⏱️  Total time: {processing_time:.1f}ms")
                print(f"🔧 API processing time: {api_time}ms")
                
                # Calculate texts per second safely
                if api_time > 0:
                    texts_per_sec = len(texts) / (api_time / 1000)
                    print(f"📈 Texts per second: {texts_per_sec:.1f}")
                else:
                    print(f"📈 Texts per second: N/A (API time = 0)")
                
                if cache_info:
                    print(f"💾 Cache info: {cache_info}")
                    
                # Show first few results
                lemmas = result.get('lemmas', [])
                print(f"📝 First 3 results:")
                for i, lemma in enumerate(lemmas[:3]):
                    print(f"   {i+1}. '{texts[i]}' → '{lemma}'")
                    
            else:
                print(f"❌ Error: {response.status}")
                error_text = await response.text()
                print(f"   {error_text}")
                
    except Exception as e:
        print(f"❌ Exception: {e}")

async def run_performance_tests():
    """Run comprehensive performance tests"""
    print("🚀 Turkish Lemmatizer Performance Tests")
    print("=" * 50)
    
    # Test different scenarios
    test_scenarios = [
        (TEST_DATA[:1], "Single Text Test"),
        (TEST_DATA[:5], "Small Batch Test (5 texts)"),
        (TEST_DATA, "Medium Batch Test (20 texts)"),
        (generate_large_dataset(TEST_DATA, 10), "Large Batch Test (200 texts)"),
        (generate_large_dataset(TEST_DATA, 50), "Very Large Batch Test (1000 texts)"),
        (generate_large_dataset(TEST_DATA, 100), "Massive Batch Test (2000 texts)"),
    ]
    
    async with aiohttp.ClientSession() as session:
        # Test service health first
        try:
            async with session.get('http://localhost:8077/health') as response:
                if response.status == 200:
                    print("✅ Service is healthy")
                else:
                    print("❌ Service health check failed")
                    return
        except Exception as e:
            print(f"❌ Cannot connect to service: {e}")
            print("💡 Make sure the service is running on http://localhost:8077")
            return
        
        # Run performance tests
        for texts, test_name in test_scenarios:
            await test_performance(session, texts, test_name)
            
        # Test caching effectiveness
        print(f"\n🔄 Cache Effectiveness Test")
        print("Testing same data twice to measure cache hits...")
        
        cache_test_data = TEST_DATA[:10]
        
        # First run (cold cache)
        await test_performance(session, cache_test_data, "Cache Test - First Run")
        
        # Second run (warm cache)
        await test_performance(session, cache_test_data, "Cache Test - Second Run")

def print_performance_recommendations():
    """Print performance recommendations"""
    print("\n" + "=" * 60)
    print("📋 PERFORMANCE RECOMMENDATIONS")
    print("=" * 60)
    
    print("\n🎯 Optimization Strategies Implemented:")
    print("   ✅ Batch Processing: Texts processed in groups of 500")
    print("   ✅ Parallel Execution: Up to 4 concurrent workers")
    print("   ✅ LRU Caching: 10,000 most recent lemmatizations cached")
    print("   ✅ Async Processing: Non-blocking I/O operations")
    print("   ✅ Smart Routing: Different strategies based on input size")
    
    print("\n⚡ Expected Performance Improvements:")
    print("   📈 3-5x faster for repeated text patterns (cache hits)")
    print("   📈 2-3x faster for large batches (parallel processing)")
    print("   📈 Better resource utilization (async + threading)")
    print("   📈 Reduced memory overhead (batch processing)")
    
    print("\n🔧 Configuration Options:")
    print("   • BATCH_SIZE = 500 (adjust based on memory)")
    print("   • MAX_WORKERS = 4 (adjust based on CPU cores)")
    print("   • CACHE_SIZE = 10000 (adjust based on available memory)")
    
    print("\n💡 Usage Tips:")
    print("   • Send multiple texts in single request for best performance")
    print("   • Use consistent text patterns to maximize cache hits")
    print("   • Monitor cache hit rates in response metrics")
    print("   • Consider increasing workers for CPU-bound workloads")

if __name__ == "__main__":
    print("🔧 Starting performance tests...")
    print("💡 Make sure the lemmatizer service is running!")
    print("   Run: python app.py or docker-compose up")
    print()
    
    try:
        asyncio.run(run_performance_tests())
        print_performance_recommendations()
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
